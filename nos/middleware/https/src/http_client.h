/*
* Copyright (c) hozonauto. 2021-2021. All rights reserved.
* Description: Http IF class
*/

#pragma once

#include <thread>
#include <mutex>
#include "https_types.h"
#include "curl/curl.h"
#include "secure_connect.h"
#include <condition_variable>
#include "crypto_adapter.h"
#include <atomic> 
namespace hozon {
namespace netaos {
namespace https {

class HttpClient {
public:
    HttpClient();

    ~HttpClient();

    void Init();

    void Start();

    void Stop();

    bool ReStart();

    bool CancelRequest(int id);

    int HttpRequest( RequestPtr req_ptr, ResponseHandler handler);

    size_t getDownloadFileSize(int id, const std::string url);

    static bool checkDisk(int id, const std::string path, size_t fileLen);

    bool Query(std::vector<Response>& respInfo);

private:
    CURL* SetupHttpRequest(int id); // context_map_mutex_ locked is required.

    std::string MakeHeader(int type, std::string value);

    std::string MakeHeader(std::string key, std::string value);

    void NotifyResult(CURL* handle, int result);

    // curl callbacks
    static size_t WriteCallback(char *ptr, size_t size, size_t nmemb, void *userdata);

    static size_t HeaderCallback(char *ptr, size_t size, size_t nmemb, void *userdata);

    static size_t progress_callback(void *clientp, double dltotal, double dlnow, double ultotal, double ulnow);

    static CURLcode SslCtxCallback(CURL *curl, void *ssl_ctx, void *userptr);

    // static CURLcode SslCtxACCallback(CURL* curl, void* ssl_ctx, void* userptr);

        enum {
            HTTP_STATE_IDLE,
            HTTP_STATE_TO_REQUEST,
            HTTP_STATE_REQUESTING,
            HTTP_STATE_CANCEL,
            HTTP_STATE_DONE,
        };

    // std::atomic_bool suspend_;
    // std::atomic_bool cancel_;
    int req_id_;
    bool stopped_;

    std::thread work_th_;
    std::condition_variable queue_wait_cond_;

    hozon::netaos::crypto::CryptoAdapter crypto_adapter_;
};

}
}
}