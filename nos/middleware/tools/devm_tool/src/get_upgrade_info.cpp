
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <regex>

#include "get_upgrade_info.h"
#include "progressbar.hpp"
#include "devm_tool_logger.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
#include "json/json.h"


namespace hozon {
namespace netaos {
namespace tools {

std::string ecu_array[] = {"all", "orin", "lidar", "srrfl", "srrfr", "srrrl", "srrrr"};

UpgradeInfoZmq::UpgradeInfoZmq(std::vector<std::string> arguments)
        :arguments_(arguments)
{
    client_ = std::make_shared<ZmqIpcClient>();
}
UpgradeInfoZmq::~UpgradeInfoZmq()
{
}
void
UpgradeInfoZmq::PrintUsage()
{
    std::cout << "Usage: nos devm upgrade [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  status,                 Used to query status of the update manage." << std::endl;
    std::cout << "  precheck,               Used to query the pre-check result." << std::endl;
    std::cout << "  progress,               Used to query the progress." << std::endl;
    std::cout << "  update [-f] <filepath>, Used to upgrade with a specified software package." << std::endl;
    std::cout << "  update [-s] <filepath>, Used to upgrade with a same version." << std::endl;
    std::cout << "                          -f: forced upgrade." << std::endl;
    std::cout << "  version,                Used to show the current version information." << std::endl;
    std::cout << "  finish,                 Used to finsh an upgrade." << std::endl;
    std::cout << "  result,                 Used to query ecu upgrade result." << std::endl;
    std::cout << "  cur_partition,          Used to query nos current partition." << std::endl;
    std::cout << "  help,                   Used to show the Usage." << std::endl;
    std::cout << std::endl;
}
void
UpgradeInfoZmq::PrintCurTime() {
    char time_check[80];
    time_t rawtime;
    struct tm timeinfo;

    time(&rawtime);
    localtime_r(&rawtime, &timeinfo);
    strftime(time_check, sizeof(time_check), "%a %b %d %H:%M:%S %Y", &timeinfo);
    std::cout << time_check << std::endl;//time
}
int32_t
UpgradeInfoZmq::StartGetUpgradeInfo()
{
    if (arguments_.size() < 1) {
        PrintUsage();
        DEVMTOOL_ERROR << "arguments_ arguments_ number too few.";
        return -1;
    }
    
    if(arguments_[0] == "status") {
        //std::cout << "upgrade status." << std::endl;
        upgrade_status();
    }
    else if (arguments_[0] == "precheck") {
        //std::cout << "upgrade precheck." << std::endl;
        upgrade_precheck();
    }
    else if(arguments_[0] == "progress") {
        //std::cout << "upgrade progress." << std::endl;
        upgrade_progress();
    }
    else if(arguments_[0] == "update") {
        //std::cout << "upgrade update." << std::endl;
        if (arguments_.size() < 2) {
            PrintUsage();
            DEVMTOOL_ERROR << "update arguments_ number too few.";
            return -1;
        }
        arguments_.erase(arguments_.begin());
        int32_t ecu_mode = 0;
        bool skip_version = false;
        bool precheck = true;
        auto it = std::find(arguments_.begin(), arguments_.end(), "-s");
        if (it != arguments_.end()) {
            skip_version = true;
            arguments_.erase(it);
        }
        it = std::find(arguments_.begin(), arguments_.end(), "-f");
        if (it != arguments_.end()) {
            precheck = false;
            arguments_.erase(it);
        }
        it = arguments_.begin();
        while (it != arguments_.end()) {
            if (*it == "all" ||
                *it == "orin" ||
                *it == "lidar" ||
                *it == "srrfl" ||
                *it == "srrfr" ||
                *it == "srrrl" ||
                *it == "srrrr"
            ) {
                ecu_mode = *it == "all" ? 0
                    : *it == "orin" ? 1
                    : *it == "lidar" ? 2
                    : *it == "srrfl" ? 3
                    : *it == "srrfr" ? 4
                    : *it == "srrrl" ? 5
                    : *it == "srrrr" ? 6
                    : 0;
                arguments_.erase(it);
                break;
            }
            it++;
        }
        if (arguments_.size() < 1) {
            PrintUsage();
            DEVMTOOL_ERROR << "update arguments_ number too few.";
            return -1;
        }
        //std::cout << "update path " << arguments_[0] << " precheck " << precheck << " ecu_mode " << ecu_mode << " skip_version " << skip_version << std::endl;
        upgrade_update(arguments_[0], precheck, ecu_mode, skip_version);
    }
    else if(arguments_[0] == "version") {
        //std::cout << "upgrade version." << std::endl;
        upgrade_version();
    }
    else if(arguments_[0] == "finish") {
        //std::cout << "upgrade finish." << std::endl;
        upgrade_finish();
    }
    else if(arguments_[0] == "result") {
        //std::cout << "upgrade result." << std::endl;
        upgrade_result();
    }
    else if(arguments_[0] == "cur_partition") {
        //std::cout << "upgrade cur_partition." << std::endl;
        upgrade_partition();
    }
    else if(arguments_[0] == "switch_slot") {
        //std::cout << "upgrade cur_partition." << std::endl;
        upgrade_switch_slot();
    }
    else if(arguments_[0] == "help") {
        PrintUsage();
    }
    else {
        PrintUsage();
        DEVMTOOL_ERROR << "upgrade arguments_[0] err";
        return -1;
    }
    
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_status()
{
    client_->Init("tcp://localhost:11130");
    PrintCurTime();
    std::cout << std::endl;

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "status request ret err.";
        PrintCurTime();
        client_->Deinit();
        return -1;
    }

    UpgradeStatusResp status{};
    status.ParseFromString(reply);
    std::cout << "status: " << status.update_status() << std::endl;
    if (status.error_code() > 0) {
        std::cout << "Error: " << status.error_code() << ", " << status.error_msg() << std::endl;
        DEVMTOOL_WARN << "Error: " << status.error_code() << ", " << status.error_msg();
    }

    PrintCurTime();
    client_->Deinit();
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_precheck(void)
{
    client_->Init("tcp://localhost:11131");
    PrintCurTime();
    std::cout << std::endl;

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "precheck request ret err.";
        PrintCurTime();
        client_->Deinit();
        return -1;
    }

    UpgradePrecheckResp precheck{};
    precheck.ParseFromString(reply);
    if (precheck.error_code() > 0) {
        std::cout << "Error: " << precheck.error_code() << ", " << precheck.error_msg() << std::endl;
        DEVMTOOL_WARN << "Error: " << precheck.error_code() << ", " << precheck.error_msg();
    }
    else {
        if (precheck.space() == true 
            && precheck.speed() == true 
            && precheck.gear() == true) {
            std::cout << "precheck succ." << std::endl;
        }
        else {
            std::cout << "precheck fail." << std::endl;
        }
        std::cout << (precheck.space() ? "\033[32m[ OK ]\033[0m" : "\033[31m[ NG ]\033[0m") << " space" << std::endl;
        std::cout << (precheck.speed() ? "\033[32m[ OK ]\033[0m" : "\033[31m[ NG ]\033[0m") << " speed" << std::endl;
        std::cout << (precheck.gear()  ? "\033[32m[ OK ]\033[0m" : "\033[31m[ NG ]\033[0m") << " gear" << std::endl;
        std::cout << std::endl;
    }

    PrintCurTime();
    client_->Deinit();
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_progress(void)
{
    client_->Init("tcp://localhost:11132");
    PrintCurTime();
    std::cout << std::endl;

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "progress request ret err.";
        PrintCurTime();
        client_->Deinit();
        return -1;
    }

    UpgradeProgressResp progress{};
    progress.ParseFromString(reply);
    if (progress.error_code() > 0) {
        std::cout << "Error: " << progress.error_code() << ", " << progress.error_msg() << std::endl;
        DEVMTOOL_WARN << "Error: " << progress.error_code() << ", " << progress.error_msg();
    }
    else {
        std::cout << "upgrade progress: " << progress.progress() << "%" << std::endl;
    }

    PrintCurTime();
    client_->Deinit();
    return 0;
}

std::vector<std::string>
UpgradeInfoZmq::Split(const std::string& inputStr, const std::string& regexStr) {
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

int32_t
UpgradeInfoZmq::upgrade_update(const std::string package_path, bool precheck, int32_t ecu_mode, bool skip_version)
{
    PrintCurTime();
    std::cout << std::endl;

    char real_path[MAXPATHLEN]{};
    if (realpath(package_path.c_str(), real_path) == nullptr) {
        DEVMTOOL_ERROR << "update package not exist, " << package_path;
        std::cout << "Error: update package not exist, " << package_path << std::endl;
        PrintCurTime();
        return -1;
    }

    std::string cur_ver{}; 
    do {
        std::ifstream ifs("/cfg/dids/dids.json");
        if (!ifs.is_open()) {
            DEVMTOOL_ERROR << "Failed to open file.";
            break;
        }

        Json::CharReaderBuilder reader;
        Json::Value root;
        JSONCPP_STRING errs;
        bool res = Json::parseFromStream(reader, ifs, &root, &errs);
        if (!res || !errs.empty()) {
            DEVMTOOL_ERROR << "parseJson error, message: " << errs;
            break;
        }
        ifs.close();

        Json::Value kv = root["kv_vec"];
        for (const auto& key_value : kv) {
            if ("F1C0" == key_value["key"].asString()) {
                DEVMTOOL_INFO << "GetCfgValueFromFile read key: " << key_value["value"]["string"].asString();
                cur_ver = key_value["value"]["string"].asString();
            }
        }
    } while(0);

    // "EP41_ORIN_HZdev_04.02.01_1121_1213_20231218.zip" 取 "04.02.01"
    std::string file_name{};
    size_t pos = package_path.find_last_of('/');
    if (pos != std::string::npos) {
        file_name = package_path.substr(pos + 1);
    }
    else {
        file_name = package_path;
    }
    std::string target_ver{};
    std::vector<std::string> target_vec = Split(file_name, "_");
    if(target_vec.size() >= 4) {
        target_ver = target_vec[3];
    }

    DEVMTOOL_INFO << "update current version " << cur_ver <<", target version " << target_ver << ".";
    std::cout << "update current version " << cur_ver << std::endl;
    std::cout << "update target  version " << target_ver << std::endl;
    std::cout << std::endl;

    client_->Init("tcp://localhost:11133");
    UpgradeUpdateReq req{};
    req.set_start_with_precheck(precheck);
    req.set_skip_version(skip_version);
    req.set_ecu_mode(ecu_mode);
    req.set_package_path(real_path);
    std::string reply{};
    DEVMTOOL_INFO << "UpgradeUpdateReq ZMQ Request start ";
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    DEVMTOOL_INFO << "UpgradeUpdateReq ZMQ Request ret " << ret;
    // if (ret < 0) {
        // 不判断失败，update_manage进入解压大包，出现延迟应答的情况
        // std::cout << "Error: zmq request ret err." << std::endl;
        // DEVMTOOL_ERROR << "update request ret err.";
        // PrintCurTime();
        // client_->Deinit();
        // return -1;
    // }

    // UpgradeUpdateResp update{};
    // update.ParseFromString(reply);
    // if (update.error_code() > 0) {
    //     std::cout << "Error: " << update.error_code() << ", " << update.error_msg() << std::endl;
    //     DEVMTOOL_WARN << "Error: " << update.error_code() << ", " << update.error_msg();
    // }
    // else {
    //     std::cout << "start update." << std::endl;
    // }
    client_->Deinit();
    DEVMTOOL_INFO << "UpgradeUpdateReq ZMQ Deinit. ";
    sleep(1);

    std::shared_ptr<ZmqIpcClient> client2_ = std::make_shared<ZmqIpcClient>();
    client2_->Init("tcp://localhost:11132");
    static uint8_t step = 0;
    static uint16_t count = 0;
    progressbar bar(100); // 100是百分率，例如1000，需要调用bar.update()1000次才能走到100%
    bar.reset();
    bar.set_todo_char(" ");
    bar.set_done_char("█");
    bar.set_opening_bracket_char("[");
    bar.set_closing_bracket_char("]");
    bar.init_bar();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    while (true) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        UpgradeCommonReq req{};
        req.set_platform(0);
        std::string reply{};
        DEVMTOOL_INFO << "UpgradeCommonReq Request progress. ";
        ret = client2_->Request(req.SerializeAsString(), reply, 1000);
        if (ret < 0) {
            std::cout << std::endl;
            std::cout << "Error: UpgradeCommonReq ZMQ request ret err: " << ret << std::endl;
            DEVMTOOL_ERROR << "UpgradeCommonReq ZMQ request ret err." << ret;
            break;
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        long int cost = std::chrono::duration<double, std::milli>(end - begin).count();

        UpgradeProgressResp progress{};
        progress.ParseFromString(reply);
        if (progress.error_code() > 0) {
            std::cout << std::endl;
            std::cout << "Error, code: " << progress.error_code() << ", message: " << progress.error_msg() << std::endl;
            DEVMTOOL_WARN << "Error: " << progress.error_code() << ", " << progress.error_msg();
            break;
        }
        else {
            //std::cout << "update progress: " << progress.progress() << "%" << std::endl;
            switch (step) {
            case 0:
            {
                count++;
                if (progress.progress() >= 2) {
                    if (1 == ecu_mode) {
                        step = 2; // 只升级orin， 跳过step1 传感器阶段
                    } else {
                        step = 1; // 含传感器升级，进入step1 传感器阶段
                    }
                    count = 20;
                    DEVMTOOL_INFO << "[case 0:a]============== bar.update(" << count << ") ==============";
                    bar.update(count);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000 - cost));
                    DEVMTOOL_INFO << "[case 0:b]============== bar.update(" << (count / 10) + 1 << ") ==============";
                    bar.update((count / 10) + 1);
                    if (count >= 200) {
                        count = 20;
                        if (1 == ecu_mode) {
                            step = 2; // 只升级orin， 跳过step1 传感器阶段
                        } else {
                            step = 1; // 含传感器升级，进入step1 传感器阶段
                        }
                    }
                }
            }
            break;
            case 1:
            {
                if (ecu_mode >= 2 && ecu_mode <= 6) {
                    if (progress.progress() <= 20) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        DEVMTOOL_INFO << "[case 1:a]============== bar.update(" << count << ") ==============";
                    } else if (progress.progress() >= 90) {
                        bar.update(100);
                        DEVMTOOL_INFO << "[case 1:b]============== bar.update(100) ==============";
                        std::cout << "\nsensor[" << ecu_array[ecu_mode] << "] update success. it will restart automatically." << std::endl;
                        step = 0;
                        count = 0;
                        goto QUIT;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        count = progress.progress();
                        bar.update(count);
                        DEVMTOOL_INFO << "[case 1:c]============== bar.update(" << count << ") ==============";
                    }
                }
                else {
                    // 整包升级场景，正常进入的step1
                    if (progress.progress() >= 9) {
                        step = 2;
                        count = 30;
                        DEVMTOOL_INFO << "[case 1:d]============== bar.update(" << count << ") ==============";
                        bar.update(count);
                    } else {
                        count++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10000 - cost));
                        DEVMTOOL_INFO << "[case 1:e]============== bar.update(" << count << ") ==============";
                        bar.update(count);
                        if (count >= 30) {
                            step = 2;
                        }
                    }
                }
            }
            break;
            case 2:
            {
                if (0 == ecu_mode) {
                    // 整包升级场景，正常进入的step2
                    if (progress.progress() >= 9 && progress.progress() <= 20) {
                        DEVMTOOL_INFO << "[case 2:a]============== bar.update(" << count << ") ==============";
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    } else if (progress.progress() == 90) {
                        step = 0;
                        count = 0;
                        DEVMTOOL_INFO << "[case 2:b]============== bar.update(100) ==============";
                        bar.update(100);
                        std::cerr << std::endl;
                        std::cout << "update all package success, it will restart automatically." << std::endl;
                        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                        goto QUIT;
                    } else if (progress.progress() == 100) {
                        step = 0;
                        count = 0;
                        DEVMTOOL_INFO << "[case 2:c]============== bar.update(100) ==============";
                        bar.update(100);
                        std::cerr << std::endl;
                        std::cout << "sensor update success. it will restart automatically." << std::endl;
                        goto QUIT;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        count = progress.progress() + 10;
                        DEVMTOOL_INFO << "[case 2:d]============== bar.update(" << count << ") ==============";
                        bar.update(count);
                    }
                } else {
                    // 只升级orin，进入的step2，忽略传感器的10%部分
                    if (progress.progress() <= 10) {
                        DEVMTOOL_INFO << "[case 2:d]============== bar.update(" << count << ") ==============";
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    } else if (progress.progress() == 90) {
                        step = 0;
                        count = 0;
                        DEVMTOOL_INFO << "[case 2:e]============== bar.update(100) ==============";
                        bar.update(100);
                        std::cerr << std::endl;
                        std::cout << "update orin success, it will restart automatically." << std::endl;
                        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                        goto QUIT;
                    } else if (progress.progress() == 100) {
                        step = 0;
                        count = 0;
                        DEVMTOOL_INFO << "[case 2:f]============== bar.update(100) ==============";
                        bar.update(100);
                        std::cerr << std::endl;
                        std::cout << "orin version is the same and will not be upgraded. it will restart automatically." << std::endl;
                        goto QUIT;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                        count = progress.progress() + 10;
                        DEVMTOOL_INFO << "[case 2:g]============== bar.update(" << count << ") ==============";
                        bar.update(count);
                    }
                }
            }
            break;
            default:
                 break;
            }
        }
    }

QUIT:
    PrintCurTime();
    client2_->Deinit();
    DEVMTOOL_INFO << "UpgradeCommonReq ZMQ Deinit. ";
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_version(void)
{
    client_->Init("tcp://localhost:11134");
    PrintCurTime();
    std::cout << std::endl;

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "version request ret err.";
        PrintCurTime();
        client_->Deinit();
        return -1;
    }

    UpgradeVersionResp version{};
    version.ParseFromString(reply);
    if (version.error_code() > 0) {
        std::cout << "Error: " << version.error_code() << ", " << version.error_msg() << std::endl;
        DEVMTOOL_WARN << "Error: " << version.error_code() << ", " << version.error_msg();
    }
    else {
        std::vector<std::pair<std::string, std::string>> vec_version;
        vec_version.push_back(std::make_pair("major", version.major_version()));
        vec_version.push_back(std::make_pair("soc", version.soc_version()));
        vec_version.push_back(std::make_pair("mcu", version.mcu_version()));
        vec_version.push_back(std::make_pair("dsv", version.dsv_version()));
        vec_version.push_back(std::make_pair("switch", version.swt_version()));
        for (const auto& pair : version.sensor_version()) {
            vec_version.push_back(pair);
        }
        printf("%-20s%-20s\n", "Device", "Version");
        printf("%-20s%-20s\n", "----------", "----------");
        for (auto& pair : vec_version) {
            if (pair.second.length() == 0) {
                pair.second = "null";
            }
            printf("%-20s%-20s\n", pair.first.c_str(), pair.second.c_str());
        }
        printf("\n");
    }

    PrintCurTime();
    client_->Deinit();
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_finish(void)
{
    client_->Init("tcp://localhost:11135");
    PrintCurTime();
    std::cout << std::endl;

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 10000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "finish request ret err.";
        PrintCurTime();
        client_->Deinit();
        return -1;
    }

    UpgradeFinishResp finish{};
    finish.ParseFromString(reply);
    if (finish.error_code() > 0) {
        std::cout << "Error: " << finish.error_code() << ", " << finish.error_msg() << std::endl;
        DEVMTOOL_WARN << "Error: " << finish.error_code() << ", " << finish.error_msg();
    }
    else {
        std::cout << "upgrade finish success." << std::endl;
    }

    PrintCurTime();
    client_->Deinit();
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_result(void)
{
#define OTA_RESULT_FILE "/opt/usr/log/ota_log/ota_result.json"
    PrintCurTime();
    std::cout << std::endl;

    std::ifstream ifs(OTA_RESULT_FILE);
    if (!ifs.is_open()) {
        std::cerr << "Error: open file fail, " << OTA_RESULT_FILE << std::endl;
        return -1;
    }

    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if (!res || !errs.empty()) {
        std::cout << "Error: Json::parseFromStream error, " << errs << std::endl;
        DEVMTOOL_ERROR << "parseJson error, message: " << errs;
        return -1;
    }
    ifs.close();
    std::string orin = root["ORIN"].asString();
    std::string lidar = root["LIDAR"].asString();
    std::string srrfl = root["SRR_FL"].asString();
    std::string srrfr = root["SRR_FR"].asString();
    std::string srrrl = root["SRR_RL"].asString();
    std::string srrrr = root["SRR_RR"].asString();

    printf("%-20s%-20s\n", "Device", "status");
    printf("%-20s%-20s\n", "----------", "----------");
    printf("%-20s%-20s\n", "ORIN", orin.size() == 0 ? "null" : orin.c_str());
    printf("%-20s%-20s\n", "LIDAR", lidar.size() == 0 ? "null" : lidar.c_str());
    printf("%-20s%-20s\n", "SRR_FL", srrfl.size() == 0 ? "null" : srrfl.c_str());
    printf("%-20s%-20s\n", "SRR_FR", srrfr.size() == 0 ? "null" : srrfr.c_str());
    printf("%-20s%-20s\n", "SRR_RL", srrrl.size() == 0 ? "null" : srrrl.c_str());
    printf("%-20s%-20s\n", "SRR_RR", srrrr.size() == 0 ? "null" : srrrr.c_str());
    printf("\n");

    PrintCurTime();
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_partition(void)
{
    client_->Init("tcp://localhost:11136");
    PrintCurTime();
    std::cout << std::endl;

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 2000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "partition request ret err.";
        PrintCurTime();
        client_->Deinit();
        return -1;
    }

    UpgradeCurPartitionResp partition{};
    partition.ParseFromString(reply);
    if (partition.error_code() > 0) {
        std::cout << "Error: " << partition.error_code() << ", " << partition.error_msg() << std::endl;
        DEVMTOOL_WARN << "Error: " << partition.error_code() << ", " << partition.error_msg();
    }
    else {
        std::cout << "current: " << partition.cur_partition() << std::endl;
    }

    PrintCurTime();
    client_->Deinit();
    return 0;
}

int32_t
UpgradeInfoZmq::upgrade_switch_slot(void)
{
    client_->Init("tcp://localhost:11137");
    PrintCurTime();
    std::cout << std::endl;

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 2000);
    if (ret < 0) {
        std::cout << "Error: zmq request ret err." << std::endl;
        DEVMTOOL_ERROR << "partition request ret err.";
        PrintCurTime();
        client_->Deinit();
        return -1;
    }

    UpgradeSwitchSlotResp partition{};
    partition.ParseFromString(reply);
    if (partition.error_code() > 0) {
        std::cout << "Error: " << partition.error_code() << ", " << partition.error_msg() << std::endl;
        DEVMTOOL_WARN << "Error: " << partition.error_code() << ", " << partition.error_msg();
    }
    else {
        std::cout << "switch succ." << std::endl;
    }

    PrintCurTime();
    client_->Deinit();
    return 0;
}

}
}
}
