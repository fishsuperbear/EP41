
/*
* Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
* Description: test-https_download_client.cpp is designed for crypto.
*/
#include <csignal>
#include <iostream>
#include <chrono>
// #include "hz_fm_agent.h"
#include "sys/stat.h"
#include "unistd.h"
#include <iostream>
#include <csignal>
#include "ota_download.h"
#include "https_logger.h"
#include "log_moudle_init.h"
#include <thread>
using namespace hozon::netaos::https;

#include "tsp_comm.h"
#include "ota_download.h"
#include "pki_request.h"
#include "config_param.h"
#include "cfg/include/config_param.h"


sig_atomic_t g_stopFlag = 0;

void INTSigHandler(int32_t num) {
    g_stopFlag = 1;
    std::cout<< "exit.num:"<<num<<std::endl;
}

hozon::netaos::cfg::ConfigParam *cfg_param_ = nullptr;
void test_RequestHdUuid_1() {
  {
    // TspComm httpsReq;

    TspComm::GetInstance().Init();
    std::future<TspComm::TspResponse> ret_uuid = TspComm::GetInstance().RequestHdUuid();
    TspComm::TspResponse ret_request = ret_uuid.get();
    std::cout << "result_code:" << ret_request.result_code << " uuid:" << ret_request.response << std::endl;

    // std::future<TspComm::TspResponse> ret_remotecfg = TspComm::GetInstance().RequestRemoteConfig();
    // ret_request = ret_remotecfg.get();
    // std::cout << "result_code:" << ret_request.result_code << " remoteconfig:" << ret_request.response << std::endl;

    // std::future<TspComm::TspResponse> ret_uptoken = TspComm::GetInstance().RequestUploadToken();
    // ret_request = ret_uptoken.get();
    // std::cout << "result_code:" << ret_request.result_code << " uploadToken:" << ret_request.response << std::endl;


    //getCarStrategiesByVin
    std::string vin = TspComm::GetInstance().GetPkiVin();
    std::cout << "get pki vin:" <<vin<<std::endl;
    std::string ecu_domain = "http://vdc.autodrive.hozonauto.com:8081/";
    std::string url_path = "/api/strategyConfig/getCarStrategiesByVin";
    std::string request_body = "{\r\n"
                               "        \"vin\": \"" + vin + "\"\r\n"
                               "}";
    TspComm::HttpsParam https_param;
    https_param.headers.insert(std::pair<std::string,std::string>("Content-Type","application/json"));
    https_param.method = HttpsMethod::HTTPS_POST;
    https_param.url = ecu_domain + url_path;
    https_param.request_body = request_body;
    
    std::future<TspComm::TspResponse> ret_https_stra = TspComm::GetInstance().RequestHttps(https_param);
    while (!g_stopFlag && (std::future_status::ready != ret_https_stra.wait_for(std::chrono::milliseconds(500))));

    ret_request = ret_https_stra.get();
    std::cout << "result_code:" << ret_request.result_code << " strategyConfig:" << ret_request.response << std::endl;



    //getLatestConfigByTriggerId
    ecu_domain = "http://vdc.autodrive.hozonauto.com:8081/";
    url_path = "/api/cdnConfig/getLatestConfigByTriggerId";
    request_body = "{\r\n"
                        "\"triggerIdList\": \"" + vin + "\"\r\n"
                    "}";
    https_param.headers.clear();
    https_param.headers.insert(std::pair<std::string,std::string>("Content-Type","application/json"));
    https_param.method = HttpsMethod::HTTPS_POST;
    https_param.url = ecu_domain + url_path;
    https_param.request_body = request_body;

    std::future<TspComm::TspResponse> ret_https_trig = TspComm::GetInstance().RequestHttps(https_param);
    while (!g_stopFlag && (std::future_status::ready != ret_https_trig.wait_for(std::chrono::milliseconds(500))))
    ret_request = ret_https_trig.get();
    std::cout << "result_code:" << ret_request.result_code << " LatestConfigByTriggerId:" << ret_request.response << std::endl;


    // std::cout<< "pki_status:"<<httpsReq.ReadPkiStatus();

    // TspComm::GetInstance().RequestRemoteConfig();
    // TspComm::GetInstance().RequestUploadToken();


    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    TspComm::GetInstance().Deinit();
  }
    std::cout << "test_RequestHdUuid_1 exit."<<std::endl;
}

void test_func1() {
    // 创建download对象，并执行下载
    hozon::netaos::https::OtaDownload download_client;
    download_client.Init();
     // 主动查询下载状态
    std::vector<Response> respInfo;
    download_client.QueryDownloadInfo(respInfo);
    RequestPtr req(new Request());
    // req.get()->url = "http://platform.autodrive.hozonauto.com:2222/test/tempLog.txt";
    req.get()->url = "http://platform.autodrive.hozonauto.com:2222/test/forDldTest.tar.gz";
    req.get()->save_file_path = "/home/zhouyuli/下载/test_0817/forDldTest.tar.gz";
    // req.get()->save_file_path = "/storage/zyl/forDldTest.tar.gz";
    // 实时回调下载状态
    download_client.Download(req, [&](int id, ResponsePtr resp_ptr) {

      // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<<std::endl;
      // // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<< " content :"<< resp_ptr.get()->content<<std::endl;
      // std::cout<< "download rate :"<< resp_ptr.get()->rate_of_download<<" download status :"<< resp_ptr.get()->status_download<<std::endl;
    });

    // 暂停下载
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    // 主动查询下载状态
    download_client.QueryDownloadInfo(respInfo);
    std::cout<<" start sleep \n";
    download_client.StopDownLoad();

    // 主动查询下载状态
    download_client.QueryDownloadInfo(respInfo);
    // for (auto resp : respInfo) {
    //    std::cout <<"QueryDownloadInfo:: id :"<< resp.id<< " resp.rate_of_download:" << resp.rate_of_download << " resp.status_download:" << resp.status_download<<std::endl;
    // }
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    // 重新开始下载
    download_client.ReStartDownLoad();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    download_client.QueryDownloadInfo(respInfo);
    while(1) {
      sleep(1);
    }
}

void test_func2 () {
    // 创建download对象，并执行下载
    hozon::netaos::https::OtaDownload download_client;
    download_client.Init();
    RequestPtr req(new Request());
    req.get()->url = "http://platform.autodrive.hozonauto.com:2222/test/tempLog.txt";
    // req.get()->save_file_path = "/home/zhouyuli/下载/test_0817/tempLog.log";

    req.get()->save_file_path = "/storage/zyl/tempLog.log";

    download_client.Download(req, [&](int id, ResponsePtr resp_ptr) {

        // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<<std::endl;
        // // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<< " content :"<< resp_ptr.get()->content<<std::endl;
        // std::cout<< "download rate :"<< resp_ptr.get()->rate_of_download<<" download status :"<< resp_ptr.get()->status_download<<std::endl;
    });
    // 取消下载
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::cout<<" start sleep \n";
    download_client.CancelDownLoad();
}

void test_func3 () {
    // 创建download对象，并执行下载
    hozon::netaos::https::OtaDownload download_client;
    download_client.Init();
    RequestPtr req(new Request());
    req.get()->url = "http://platform.autodrive.hozonauto.com:2222/test/tempLog.txt";
    req.get()->save_file_path = "/home/zhouyuli/下载/test_0817/tempLog.log";
    download_client.Download(req, [&](int id, ResponsePtr resp_ptr) {
        // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<<std::endl;
        // // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<< " content :"<< resp_ptr.get()->content<<std::endl;
        // std::cout<< "download rate :"<< resp_ptr.get()->rate_of_download<<" download status :"<< resp_ptr.get()->status_download<<std::endl;
    });
    // 取消下载
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    download_client.CancelDownLoad();

    sleep(10);
}

void test_func4 () {
    // 断点续传
    hozon::netaos::https::OtaDownload download_client;
    download_client.Init();
    RequestPtr req(new Request());
    req.get()->url = "http://platform.autodrive.hozonauto.com:2222/test/tempLog.txt";
    req.get()->save_file_path = "/home/zhouyuli/下载/test_0817/tempLog.log";
    download_client.Download(req, [&](int id, ResponsePtr resp_ptr) {
        // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<<std::endl;
        // // std::cout <<"id :"<<id <<" code :"<< resp_ptr.get()->code<< " content :"<< resp_ptr.get()->content<<std::endl;
        // std::cout<< "download rate :"<< resp_ptr.get()->rate_of_download<<" download status :"<< resp_ptr.get()->status_download<<std::endl;
    });
    sleep(3);
}

void test_func5() {
    hozon::netaos::https::OtaDownload download_client;
    std::map<std::string, std::string> check_param{{"unzip_path", "/mnt/unzippath"}};
    std::string ota_file = "/mnt/EP41_ORIN_HZdev_08.03.09_1101_1102_20231113.tar.lrz";
    download_client.SetParam(check_param);
    download_client.Verify(ota_file);
    sleep(3);
}

int main() {

    signal(SIGTERM, INTSigHandler);
    signal(SIGINT, INTSigHandler);
    signal(SIGKILL, INTSigHandler);
    signal(SIGPIPE, SIG_IGN);

    // auto test_th = std::thread(test_RequestHdUuid_1);
    HttpsLogger::GetInstance().setLogLevel(static_cast<int32_t>(
        HttpsLogger::HttpsLogLevelType::HTTPSLOG_DEBUG));
    HttpsLogger::GetInstance().InitLogging(
        "https_TEST",       // the id of application
        "https test",  // the log id of application
        HttpsLogger::HttpsLogLevelType::
            HTTPSLOG_DEBUG,  // the log
                              // level of
                              // application
        hozon::netaos::log::HZ_LOG2CONSOLE |
            hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
        "./",  // the log file directory, active when output log to file
        10,    // the max number log file , active when output log to file
        20     // the max size of each  log file , active when output log to file
    );
    HttpsLogger::GetInstance().CreateLogger("https_TEST");
    std::cout << "main test begin" << std::endl;

    cfg_param_ = hozon::netaos::cfg::ConfigParam::Instance();
    auto res = cfg_param_->Init(3000);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        std::cout << "cfg ConfigParam Init error.ret:" << res<<std::endl;
    }

    test_RequestHdUuid_1();
    // test_func1();
    // test_func2();

    // test_func3 + test_func4 断点续传
    // test_func3();
    // test_func4();
    // test_func5();

    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    cfg_param_->DeInit();
    std::cout << "test process end  \n";
    std::cout << "main test end。" << std::endl;
    return 0;
}
