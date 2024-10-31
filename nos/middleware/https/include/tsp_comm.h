/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: https.h is designed for https.
 */
#pragma once

#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <list>
#include <atomic>

#include "https/include/https.h"
#include "https/include/https_types.h"

namespace hozon {
namespace netaos {
namespace https {

class TspComm {
   public:
    // explicit HttpsDownloadClient(std::map<std::string, std::string> param_map);
    enum ReqType:uint32_t{
        kReqHdUuid = 0,
        kReqUploadToken,
        kReqUploadRemoteConfig,
        kReqHttps,
        KReqMax
    };

    struct TspResponse {
        int32_t result_code = HttpsResult_HttpsComError;
        std::string response;
        std::string content_type;
    };

    struct HttpsParam {
        int32_t method;
        std::string url;
        std::string request_body;
        std::map<std::string, std::string> headers;
        HttpsParam() : method(0), url(""), request_body("") {}
    };

    static TspComm& GetInstance() {
        static TspComm instance;
        return instance;
    }

    void Init();
    void Deinit();
    std::string GetPkiVin(){
        return vin_;
    };
    std::future<TspResponse> RequestHdUuid();
    std::future<TspResponse> RequestUploadToken();
    std::future<TspResponse> RequestRemoteConfig();
    std::future<TspResponse> RequestHttps(HttpsParam& https_param);
private:
    TspComm();
    ~TspComm();
    bool GetPkiConfig();

    using Task = std::function<void(void)>;
    using TaskQueue = std::list<Task>;

    struct ThreadInfo {
        TaskQueue task_queue;
        std::mutex task_queue_mutex;
        std::thread th;
    };
    HttpsResultCode Communication(std::shared_ptr<https::Request> request, std::string& response_body);
    TspResponse ParseHttpBodyInjson(ReqType req_typt,std::string response_body);
    std::shared_ptr<ThreadInfo> GetThreadInfo(ReqType type);

    std::map<uint32_t, std::shared_ptr<ThreadInfo>> th_map_;
    std::mutex th_map_mutex_;
    bool stopped_ = false;
    bool inited_ = false;
    static std::atomic_bool initFlag_;
    std::mutex init_mutex_;

    std::string slot_cfg_;
    std::string vin_;
    std::string root_ca_path_;
    std::string device_cert_path_;
    uint8_t device_cert_state_;

};

}  // namespace https
}  // namespace netaos
}  // namespace hozon