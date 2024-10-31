#include <condition_variable>
#include <mutex>
#include <thread>
#include "recorder.h"

//recorder
bool is_stop_ok = false;
std::mutex recorder_condition_mtx;
std::condition_variable recorder_cv;

int main(int argc, char** argv) {

    hozon::netaos::bag::Recorder* recorder = new hozon::netaos::bag::Recorder();

    // std::thread recThread = std::thread([recorder] {           //新起线程
    //     std::this_thread::sleep_for(std::chrono::seconds(5));  //5s后调用Stop停止录制
    //     if (nullptr != recorder) {
    //         recorder->Stop();
    //         is_stop_ok = true;
    //     }
    //     std::lock_guard<std::mutex> lock(recorder_condition_mtx);
    //     recorder_cv.notify_all();
    //     std::cout << "stop ok " << std::endl;
    // });

    hozon::netaos::bag::RecordOptions rec_options;
    rec_options.output_file_name = "my_record";  //输出文件名
    // rec_options.topics.push_back("camer80");     //需要录制的topic名
    recorder->Start(rec_options);  //开始录制
                                   // std::unique_lock<std::mutex> lck(recorder_condition_mtx);  //等待Stop()被调用
                                   // recorder_cv.wait(lck, [&]() { return is_stop_ok; });

    std::this_thread::sleep_for(std::chrono::seconds(5));  //5s后调用Stop停止录制
    if (nullptr != recorder) {
        recorder->Stop();
    }

    if (nullptr != recorder) {
        delete recorder;  //释放资源
        recorder = nullptr;
    }

    std::cout << " start 2" << std::endl;

    recorder = new hozon::netaos::bag::Recorder();
    // std::this_thread::sleep_for(std::chrono::seconds(3));  //3s继续录制
    std::cout << " start 3" << std::endl;
    recorder->Start(rec_options);

    std::this_thread::sleep_for(std::chrono::seconds(3));  //5s调用Stop停止录制
    if (nullptr != recorder) {
        recorder->Stop();
    }

    std::cout << " stop 2" << std::endl;

    if (nullptr != recorder) {
        delete recorder;  //释放资源
        recorder = nullptr;
    }

    // recThread.join();
}