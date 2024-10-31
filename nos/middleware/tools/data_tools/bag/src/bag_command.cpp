#include <pthread.h>
#include <signal.h>
#include <termio.h>
#include <condition_variable>
#include <mutex>
#include "argvparser.h"
#include "attachment.h"
#include "bag_info.h"
#include "convert.h"
#include "data_tools_logger.hpp"
#include "impl/stat_impl.h"
#include "player.h"
#include "recorder.h"
#include "save.h"

using namespace hozon::netaos::bag;
using namespace std;
using namespace argvparser;

Recorder* recorder = nullptr;
Player* player = nullptr;
Convert* convert = nullptr;
int type = 0;
bool is_stop = false;
bool is_pause = false;
struct termios stored_settings;
bool has_stored_settings = false;
//recorder
bool is_stop_ok = false;
std::mutex recorder_condition_mtx;
std::condition_variable recorder_cv;

bool is_key_thread = false;
int seek_time = 0;
std::chrono::steady_clock::time_point key_down_time;

void keyboardSignalHandler(int signum) {
    tcsetattr(0, TCSANOW, &stored_settings);
    exit(0);
}

void* countThread(void* arg) {
    player->Pause();
    while (is_key_thread) {
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        if (currentTime > key_down_time) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - key_down_time) > std::chrono::milliseconds(1000)) {
                if (seek_time < 0) {
                    //快退
                    player->Rewind(std::abs(seek_time));
                    seek_time = 0;
                } else {
                    //快进
                    player->FastForward(seek_time);
                    seek_time = 0;
                }
                is_key_thread = false;
                player->Resume();
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return nullptr;
}

void* scanKeyboard(void* arg) {
    // signal(SIGINT, keyboardSignalHandler);
    struct termios new_settings;
    tcgetattr(0, &stored_settings);
    has_stored_settings = true;
    new_settings = stored_settings;
    new_settings.c_lflag &= (~ICANON);
    new_settings.c_lflag &= ~ECHO;
    new_settings.c_cc[VTIME] = 0;  //等待时间
    tcgetattr(0, &stored_settings);
    new_settings.c_cc[VMIN] = 1;  //等待的字符数
    tcsetattr(0, TCSANOW, &new_settings);
    pthread_t count_thread;
    // pthread_testcancel();
    while (!is_stop) {
        int input = -1;
        input = getchar();
        if (32 == input && nullptr != player) {
            if (is_pause) {  // space key
                player->Resume();
                is_pause = false;
            } else {
                player->Pause();
                is_pause = true;
            }
        } else if (68 == input) {  //"<-"
            if (!is_key_thread) {
                is_key_thread = true;
                pthread_create(&count_thread, NULL, countThread, NULL);
            }
            key_down_time = std::chrono::steady_clock::now();
            seek_time--;
            std::cout << "                                           \r"
                      << "jump " << seek_time << "s\r";
        } else if (67 == input) {  //"->"
            if (!is_key_thread) {
                is_key_thread = true;
                pthread_create(&count_thread, NULL, countThread, NULL);
            }
            key_down_time = std::chrono::steady_clock::now();
            seek_time++;
            std::cout << "                                           \r"
                      << "jump " << seek_time << "s\r";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    tcsetattr(0, TCSANOW, &stored_settings);
    pthread_join(count_thread, NULL);  //等待线程结束
    return nullptr;
}

// 定义信号处理程序
void signalHandler(int signum) {
    if (has_stored_settings) {
        tcsetattr(0, TCSANOW, &stored_settings);
    }
    is_stop = true;
    // 终止程序
    if (type == 1) {
        std::unique_lock<std::mutex> lck(recorder_condition_mtx);
        if (nullptr != recorder && false == is_stop_ok) {
            recorder->Stop();
            is_stop_ok = true;
            recorder_cv.notify_all();
        }
    } else if (type == 2) {
        if (nullptr != player) {
            player->Stop();
        }
    } else if (type == 3) {
        if (nullptr != convert) {
            convert->Stop();
        }
    }

    if (has_stored_settings) {
        tcsetattr(0, TCSANOW, &stored_settings);
    }
}

void* recorder_auto_exit(void* arg) {
    double exit_duration = *((double*)arg);
    auto start = std::chrono::steady_clock::now();  // 记录开始时间
    while (!is_stop_ok) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() > exit_duration) {
            signalHandler(0);
            break;
        }
    }
    return nullptr;
}

int main(int argc, char** argv) {
    hozon::netaos::data_tool_common::BAGCONSOLELogger::GetInstance().setLogLevel(hozon::netaos::data_tool_common::LogLevel::kWarn);
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    // Log::SetVerbosity(Log::Info);
    std::string log_level = "";
    bool record_show_help_info = false;
    bool play_show_help_info = false;
    bool convert_show_help_info = false;
    bool save_show_help_info = false;
    bool attachment_show_help_info = false;
    struct RecordOptions rec_options;
    double recorder_run_time = 0;
    auto record_argv_parser =
        (command("record"), option("-h", "--help").set(record_show_help_info, true) % "print the help infomation",
         option("-a", "--all").set(rec_options.record_all, true) % "record all topics" | values("topics", rec_options.topics) % "record specified topics",
         (option("-k", "--exclude-topics") & values("exclude-topics", rec_options.exclude_topics)) % "not record the specified topic.",
         option("-m", "--method").set(rec_options.method, true) % " record method topics",
         (option("--split-size") & value("split-size", rec_options.max_bagfile_size)) % "[1-65535MB]record multiple files of the same size. Default 2GB.",
         (option("--split-duration") & value("split-duration", rec_options.max_bagfile_duration)) %
             "[5-65535 s]spicify the recording time. recording to new segment file after the specified time is exceeded. Default 1h.",
         (option("-o", "--output-name") & value("output-file-name", rec_options.output_file_name)) % "specify the bag file name. Default YYYY-mm-dd_HH-MM-SS",
         option("--no-time-suffix").set(rec_options.use_time_suffix, false) % "don't add time suffix after bag name",
         (option("--format") & value("format-type", rec_options.storage_id)) % "[mcap/cyber] specify the recording format type. Default mcap",
         (option("-r", "--runtime") & value("time", recorder_run_time)) % "Set the recording runtime, and the recording will automatically end when the specified time is reached. Unit: seconds",
         (option("--queue-size") & value("queue-size", rec_options.queue_size)) % "[1-1000000] set the record message queue size. Default 1000",
         (option("--max-files") & value("max-file-num", rec_options.max_files)) %
             "specify the max file number in a folder, when the max file number is exceeded, delete the oldest files. value rang [1, 864000]. Defaul 864000.",
         (option("--max_cache_size") & value("max_cache_size", rec_options.max_cache_size)) % "max cache size",
         (option("--attachments") & values("attachments", rec_options.attachments)) % "record attachments.",
         (option("--log-level") & value("log-level", log_level)) % "set log level: [kDebug, kInfo, kWarn, kError]. Defaul 'kError'.");

    struct PlayerOptions play_options;
    auto play_argv_parser =
        (command("play"), option("-h", "--help").set(play_show_help_info, true) % "print the help infomation" | values("bag_file_name", play_options.uri) % "bag file name",
         option("-a", "--all").set(play_options.play_all_topic, true) % "play all topics", (option("-t", "--topics") & values("topic names", play_options.topics)) % "specify the topic names.",
         (option("-k", "--exclude-topics") & values("exclude-topics", play_options.exclude_topics)) % "not play the specified topic.",
         option("-l", "--loop").set(play_options.loop, true) % "unlimited loop playback",
         option("-f", "--force").set(play_options.force, true) % "if an existing node publishes the same topic, only a warning message is printed. Force playback of data.",
         (option("-s", "--start_offset") & value("start_offset", play_options.start_offset)) % "[0-10000] publish message from a certain point in time. Unit: seconds. Default 0.",
         (option("--queue-size") & value("queue-size", play_options.queue_size)) % "[1-1000] set the publish message queue size",
         (option("-r", "--rate") & value("rate", play_options.rate)) % "[0.0625-50] speify the playback rate. Default is 1(original rate)",
         (option("-d", "--delay") & value("seconds", play_options.delay)) % "play delayed n seconds. Default is 1",
         (option("-b", "--begin") & value("%Y-%m-%d-%H:%M:%S", play_options.begin)) % "play the record begin at. for example: 2018-07-01-00:00:00",
         (option("-e", "--end") & value("%Y-%m-%d-%H:%M:%S", play_options.end)) % "play the record end at. for example: 2018-07-01-00:00:00",
         option("--protomethod").set(play_options.protomethod, true) % "play topic in protomethod type.",
         (option("-c", "--lidar_conf_path") & value("lidar_conf_path", play_options.lidar_conf_path)) %
             "specify lidar correction config file path and lidar extrinsic file path. it should be directory. for example: /app/conf/ .",
         option("--no-motion-comp").set(play_options.force_play_pointcloud, true) % "Force play pointcloud, no mater whether extrinsics file exist.",
         (option("--h265") & value("<4,7,11,c11,c7,c4>", play_options.h265)) %
             "convert h265 to yuv and play. Defaul 0. 0:don't play yuv; 4:4-way surround view; 7:2 forward views, 4 peripheral views, and 1 tail view; 11:all view",
         (option("--log-level") & value("log-level", log_level)) % "set log level: [kDebug, kInfo, kWarn, kError]. Defaul 'kError'.",
         (option("-u", "--update-pubtime").set(play_options.update_pubtime, true) % "set update publish_stamp flag"));

    struct ConvertOptions convert_options;
    auto convert_argv_parser =
        (command("convert"), option("-h", "--help").set(convert_show_help_info, true) % "print the help infomation",
         (option("-i", "--input-file") & value("input-file", convert_options.input_file)) % "input bag file name.",
         (option("--input-file-type") & value("input-file-type", convert_options.intput_file_type)) % "data type of input file [rtfbag/mcap]. Defaul 'rtfbag'.",
         (option("--input-data-version") & value("input-data-version", convert_options.intput_data_version)) % "data version of input file [0228-0324/0430-0613/0430-0723]. Defaul:'0228-0324'.",
         (option("-o", "--out-file") & value("out-file", convert_options.output_file)) % "out put bag file name.",
         (option("--out-file-type") & value("out-file-type", convert_options.output_file_type)) % "data type of out file [mcap/cyber]. Defaul 'mcap'.",
         (option("--out-data-version") & value("out-data-version", convert_options.output_data_version)) % "data version of out file. Defaul:'0000-0001'.",
         (option("-t", "--topics") & value("topics name", convert_options.topics)) % "convert the specified topics.",
         (option("-k", "--exclude-topics") & values("exclude-topics", convert_options.exclude_topics)) % "not convert the specified topics.",
         option("--no-time-suffix").set(convert_options.use_time_suffix, false) % "don't add time suffix after bag name",
         (option("--log-level") & value("log-level", log_level)) % "set log level: [kDebug, kInfo, kWarn, kError]. Defaul 'kError'.");

    struct InfoOptions info_options;
    auto info_argv_parser =
        (command("info"), option("-h", "--help").set(info_options.show_help_info, true) % "print the help infomation" | values("file-path", info_options.bag_files) % "bag files path");

    struct SaveOptions save_options;
    auto save_argv_parser = (command("save"), option("-h", "--help").set(save_show_help_info, true) % "print the help infomation" | values("file-path", save_options.url) % "bag files path",
                             (option("-t", "--topics") & values("topic names", save_options.topics)) % "specify the topic names.");

    struct StatImplOptions stat_options;
    bool check_show_help_info = false;
    auto check_seq_argv_parser =
        (command("stat"), (command("check_seq"), option("-h", "--help").set(check_show_help_info, true) % "print the help infomation" | values("file", stat_options.url) % "mcap file path.",
                           (option("-t", "--topic") & values("time name", stat_options.topics_list)) % "topic name which need to be check."));

    struct AttachmentOptions attachment_options;
    auto attachment_argv_parser =
        (command("attachment"), option("-h", "--help").set(attachment_show_help_info, true) % "print the help infomation" | values("file", attachment_options.url) % "bag file path.",
         (option("-w", "--write").set(attachment_options.rewrite_attachment_, true) & values("attachment write", attachment_options.rewrite_list_)) %
             "write attachenmt to bag with custom name. The format must be like :\n    <attachment path>:<attachment name in bag>\n if the <attachment name in bag> is already exist in bag, you can use -f to "
             "overwrite.",
         // (option("-a", "--add").set(attachment_options.add_new_attachment_, true) & values("add new attachment", attachment_options.add_list_)) % "add new attachment to an exist bag",
         (option("-s", "--show").set(attachment_options.show_attachment_, true) & values("show attachment in bag", attachment_options.show_list_)) % "show attachment with name in bag",
         (option("-e", "--extract").set(attachment_options.extract_attachment_, true) & values("extract attachment in bag", attachment_options.extract_list_)) % "extract attachment from bag, format like : \n    <attachment_name_in_bag>\nor :\n    <attachment_name_in_bag>:<attachment_path_output>",
         (option("-o") & values("custom output bag name", attachment_options.rewrite_name_)) % "custom output bag name",
         (option("-f").set(attachment_options.force_write_, true)) % "force overwrite attachment");

    // auto attachment_argv_parser =
    //     (command("attachment"),
    //     (command("rewrite"), option("-h", "--help").set(attachment_show_help_info, true) % "print the help infomation" | values("file", attachment_options.url)  % "mcap file path."),
    //     (command("rewrite"), option("-h", "--help").set(attachment_show_help_info, true) % "print the help infomation" | values("file", attachment_options.url)  % "mcap file path."),
    //      option("-h", "--help").set(attachment_show_help_info, true) % "print the help infomation" | values("file", attachment_options.url) % "mcap file path.",
    //     (option("-r", "--rewrite").set(attachment_options.rewrite_attachment_, true) & values("attachment rewrite", attachment_options.rewrite_list_)) % "rewrite attachenmt in mcap, The format must be like <attachment path>:<attachments name in bag>",
    //     (option("-a", "--add").set(attachment_options.add_new_attachment_, true) & values("add new attachment", attachment_options.add_list_)) % "add new attachment to an exist bag");

    auto hztool_argv_parser = (record_argv_parser | play_argv_parser | info_argv_parser | convert_argv_parser | save_argv_parser | check_seq_argv_parser | attachment_argv_parser);
    if (parse(argc, argv, hztool_argv_parser)) {

        hozon::netaos::log::LogLevel bag_tool_level = hozon::netaos::log::LogLevel::kError;
        if ("" != log_level) {
            if ("kDebug" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kDebug;
            } else if ("kInfo" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kInfo;
            } else if ("kWarn" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kWarn;
            } else if ("kError" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kError;
            }
        }
        hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().InitLogg(bag_tool_level);
        if (record_show_help_info) {
            cout << make_man_page(record_argv_parser, argv[0]) << '\n';
        } else if (play_show_help_info) {
            cout << make_man_page(play_argv_parser, argv[0]) << '\n';
        } else if (info_options.show_help_info) {
            cout << make_man_page(info_argv_parser, argv[0]) << '\n';
        } else if (convert_show_help_info) {
            cout << make_man_page(convert_argv_parser, argv[0]) << '\n';
        } else if (save_show_help_info) {
            cout << make_man_page(save_argv_parser, argv[0]) << '\n';
        } else if (check_show_help_info) {
            cout << make_man_page(check_seq_argv_parser, argv[0]) << '\n';
        } else if (attachment_show_help_info) {
            cout << make_man_page(attachment_argv_parser, argv[0]) << '\n';
        } else if (info_options.bag_files.size() > 0) {
            ShowBagInfo(info_options);
        } else if ("" != play_options.uri) {
            pthread_t thread;
            int result = pthread_create(&thread, NULL, scanKeyboard, NULL);
            type = 2;
            player = new Player();
            std::cout << "\nHit 'space' to toggle paused, '->' to fast forward, '<-' to rewind.\n" << std::endl;
            player->Start(play_options);
            delete player;

            result = pthread_cancel(thread);  // 取消子线程
            if (result != 0) {
                std::cerr << "Failed to cancel thread: " << result << std::endl;
                return 1;
            }
            pthread_join(thread, NULL);  //等待线程结束
            tcsetattr(0, TCSANOW, &stored_settings);
        } else if (rec_options.record_all || rec_options.topics.size() > 0 || !rec_options.exclude_topics.empty()) {
            type = 1;
            recorder = new Recorder();
            if ("cyber" == rec_options.storage_id) {
                rec_options.storage_id = "record";
            }

            recorder->Start(rec_options);
            pthread_t thread;

            if (0 != recorder_run_time) {
                int result = pthread_create(&thread, NULL, recorder_auto_exit, &recorder_run_time);
            }
            {
                std::unique_lock<std::mutex> lck(recorder_condition_mtx);
                recorder_cv.wait(lck, [&]() { return is_stop_ok; });
                if (nullptr != recorder) {
                    delete recorder;
                    recorder = nullptr;
                }
            }
            if (0 != recorder_run_time) {
                pthread_join(thread, NULL);  //等待线程结束
            }

        } else if (convert_options.input_file != "") {
            type = 3;
            convert = new Convert();
            convert->Start(convert_options);
            delete convert;
        } else if (save_options.topics.size() > 0) {
            Save save;
            save.Start(save_options);
        } else if ("" != stat_options.url) {
            StatImplmpl stat_obj;
            stat_obj.Start(stat_options);
        } else if ("" != attachment_options.url) {
            Attachment attachment;
            attachment.Start(attachment_options);
        } else {
            cout << "Parameter error, use - h to print the help infomation" << '\n';
        }
    } else {
        cout << make_man_page(hztool_argv_parser, argv[0]) << '\n';
    }
    return 0;
}