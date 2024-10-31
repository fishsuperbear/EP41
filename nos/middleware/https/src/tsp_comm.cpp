#include "tsp_comm.h"

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include "json/json.h"

#include "cfg/include/config_param.h"
#include "https_logger.h"
#include "log_moudle_init.h"

namespace hozon {
namespace netaos {
namespace https {

void TspComm::Init() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (inited_) {
        return;
    }

    std::cout  << "HttpsRequest construct" << std::endl;
    if (LogModuleInit::getInstance()) {
        LogModuleInit::getInstance()->initLog();
    }

    if (!GetPkiConfig()) {
        HTTPS_ERROR << "get pki config failed";
        return;
    }

    Https::Instance().Init();
    inited_ = true;
}

void TspComm::Deinit() {
    HTTPS_WARN << "TspComm Deinit";
    inited_ = false;
    stopped_ = true;
    for(uint32_t i = ReqType::kReqHdUuid;i<ReqType::KReqMax;i++){
        if (th_map_[i]) {
            std::cout <<"th_map_["<<i<<"] is exsit."<<std::endl;
            if (th_map_[i]->th.joinable()) {
                th_map_[i]->th.join();
                std::cout << "th_map[" << i << "] joined."<<std::endl;
            }
        }
    }
}

TspComm::TspComm() {

}

TspComm::~TspComm() {
    std::cout << "HttpsRequest destructed."<<std::endl;
    if (inited_) {
        Deinit();
    }
}

std::future<TspComm::TspResponse> TspComm::RequestHdUuid() {

    std::shared_ptr<std::promise<TspComm::TspResponse>> promise_uuid = std::make_shared<std::promise<TspComm::TspResponse>>();
    if (!inited_) {
        HTTPS_ERROR << "lack of pki config from config server.";
        promise_uuid->set_value({HttpsResultCode::HttpsResult_InitError, "", ""});
        return promise_uuid->get_future();
    }

    std::shared_ptr<ThreadInfo> th_info = GetThreadInfo(kReqHdUuid);

    Task task = [promise_uuid, this]() {
        Json::Value root;
        // root["vin"] = "LUZATEST257706474";  //TODO get from hz_tsp_pki
        root["vin"] = vin_;
        std::string ecu_domain = "https://adcsapi-pki-uat.carobo.cn:18444";
        std::string url_path = "/pivot/mds-api/virualVin/1.0/virtualVin";
        std::string request_body = root.toStyledString();
        std::map<std::string, std::string> headers;
        headers["Content-Type"] = "application/json";
        std::string url = ecu_domain + url_path + "?vin=" + vin_;

        auto req_ptr = std::make_shared<https::Request>();
        req_ptr->method = HttpsMethod::HTTPS_POST;
        req_ptr->headers.emplace("Content-Type", "application/json");
        req_ptr->post_data = root.toStyledString();
        req_ptr->root_ca_cert_file = root_ca_path_;
        req_ptr->client_cert_chain = device_cert_path_;
        req_ptr->client_ap_priv_key_slot = slot_cfg_;
        req_ptr->sdkType = https::OPENSSL_AP;
        req_ptr->url = url;

        for (auto& it : headers){
            req_ptr->headers[it.first] = it.second;
        }

        std::string response_body;
        TspComm::TspResponse result_uuid;
        int32_t result_com = Communication(req_ptr, response_body);
        HTTPS_INFO << "result_com: " << result_com << "response_body:" << response_body;

        if ((result_com == static_cast<int32_t>(HttpsResultCode::HttpsResult_Success)) && response_body.size() > 0) {
            result_uuid = ParseHttpBodyInjson(ReqType::kReqHdUuid, response_body);
            // HTTPS_INFO << "uuid:" << result_uuid.response;
            promise_uuid->set_value(result_uuid);
        } else {
            result_uuid.result_code = result_com;
            HTTPS_INFO << "uuid result_code:" << result_com;
            promise_uuid->set_value(result_uuid);
        }
    };

    {
        std::lock_guard<std::mutex> lock(th_info->task_queue_mutex);
        th_info->task_queue.push_back(task);
    }

    return promise_uuid->get_future();
}

std::future<TspComm::TspResponse> TspComm::RequestUploadToken() {

    std::shared_ptr<std::promise<TspComm::TspResponse>> promise_uploadtoken = std::make_shared<std::promise<TspComm::TspResponse>>();
    if (!inited_) {
        HTTPS_ERROR << "lack of pki config from config server.";
        promise_uploadtoken->set_value({HttpsResultCode::HttpsResult_InitError, "", ""});
        return promise_uploadtoken->get_future();
    }

    std::shared_ptr<ThreadInfo> th_info = GetThreadInfo(kReqUploadToken);

    Task task = [promise_uploadtoken, this]() {

        std::string vin = vin_;
        std::string ecu_domain = "https://adcsapi-pki-uat.carobo.cn:18444";
        std::string url_path = "/pivot/file-service-api/adcsManage/applyUploadToken";
        std::string url = ecu_domain + url_path;
        std::string request_body = "{\r\n"
                                    "        \"vin\": \"" + vin + "\"\r\n"
                                    "}";
        std::map<std::string, std::string> headers;
        headers["Content-Type"] = "application/json";

        auto req_ptr = std::make_shared<https::Request>();
        req_ptr->method = HttpsMethod::HTTPS_POST;
        req_ptr->headers.emplace("Content-Type","application/json");
        req_ptr->root_ca_cert_file = root_ca_path_;
        req_ptr->client_cert_chain = device_cert_path_;
        req_ptr->client_ap_priv_key_slot = slot_cfg_;
        req_ptr->sdkType = https::OPENSSL_AP;
        req_ptr->url = url;
        req_ptr->post_data = request_body;
        for (auto& it : headers) {
            req_ptr->headers[it.first] = it.second;
        }

        std::string response_body;
        TspComm::TspResponse result_uptoken;
        int32_t result_com = Communication(req_ptr, response_body);
        if ((result_com == static_cast<int32_t>(HttpsResultCode::HttpsResult_Success)) && response_body.size() > 0) {
            result_uptoken = ParseHttpBodyInjson(ReqType::kReqUploadToken, response_body);
            // HTTPS_INFO << "uploadtoken:" << result_uptoken.response;
            promise_uploadtoken->set_value(result_uptoken);
        } else {
            result_uptoken.result_code = result_com;
            HTTPS_INFO << "uploadtoken result_code:" << result_com;
            promise_uploadtoken->set_value(result_uptoken);
        }
    };

    {
        std::lock_guard<std::mutex> lock(th_info->task_queue_mutex);
        th_info->task_queue.push_back(task);
    }

    return promise_uploadtoken->get_future();
}

std::future<TspComm::TspResponse> TspComm::RequestRemoteConfig() {
    std::shared_ptr<std::promise<TspComm::TspResponse>> promise_remoteconfig = std::make_shared<std::promise<TspComm::TspResponse>>();
    if (!inited_) {
        HTTPS_ERROR << "lack of pki config from config server.";
        promise_remoteconfig->set_value({HttpsResultCode::HttpsResult_InitError, "", ""});
        return promise_remoteconfig->get_future();
    }

    std::shared_ptr<ThreadInfo> th_info = GetThreadInfo(kReqUploadRemoteConfig);

    Task task = [promise_remoteconfig, this]() {
        std::string vin = vin_;
        std::string ecu_domain = "https://adcsapi-pki-uat.carobo.cn:18444";
        std::string url_path = "/pivot/file-service-api/adcsManage/applyUploadToken";
        std::string url = ecu_domain + url_path;
        std::string request_body = "{\r\n"
                                    "        \"vin\": \"" + vin + "\"\r\n"
                                    "}";
        std::map<std::string, std::string> headers;
        headers["Content-Type"] = "application/json";

        auto req_ptr = std::make_shared<https::Request>();
        req_ptr->method = HttpsMethod::HTTPS_POST;
        req_ptr->headers.emplace("Content-Type","application/json");
        req_ptr->root_ca_cert_file = root_ca_path_;
        req_ptr->client_cert_chain = device_cert_path_;
        req_ptr->client_ap_priv_key_slot = slot_cfg_;
        req_ptr->sdkType = https::OPENSSL_AP;
        req_ptr->url = url;
        req_ptr->post_data = request_body;
        for (auto& it : headers) {
            req_ptr->headers[it.first] = it.second;
        }

        std::string response_body;
        TspComm::TspResponse result_remoteconfig;
        int32_t result_com = Communication(req_ptr, response_body);
        if ((result_com == static_cast<int32_t>(HttpsResultCode::HttpsResult_Success)) && response_body.size() > 0) {
            result_remoteconfig = ParseHttpBodyInjson(ReqType::kReqUploadToken, response_body);
            // HTTPS_INFO << "remoteconfig:" << result_remoteconfig.response;
            promise_remoteconfig->set_value(result_remoteconfig);
        } else {
            result_remoteconfig.result_code = result_com;
            HTTPS_INFO << "remoteconfig result_code:" << result_com;
            promise_remoteconfig->set_value(result_remoteconfig);
        }
    };

    {
        std::lock_guard<std::mutex> lock(th_info->task_queue_mutex);
        th_info->task_queue.push_back(task);
    }
    return promise_remoteconfig->get_future();
}

std::future<TspComm::TspResponse> TspComm::RequestHttps(HttpsParam& https_param){
    std::shared_ptr<std::promise<TspComm::TspResponse>> promise_https = std::make_shared<std::promise<TspComm::TspResponse>>();
    if (!inited_) {
        HTTPS_ERROR << "lack of pki config from config server.";
        promise_https->set_value({HttpsResultCode::HttpsResult_InitError, "", ""});
        return promise_https->get_future();
    }

    std::shared_ptr<ThreadInfo> th_info = GetThreadInfo(ReqType::kReqHttps);
    Task task = [promise_https,&https_param, this]() {
        HTTPS_INFO << "RequestHttps task begin .";
        auto req_ptr = std::make_shared<https::Request>();
        req_ptr->method = https_param.method;
        req_ptr->url = https_param.url;
        req_ptr->post_data = https_param.request_body;
        req_ptr->root_ca_cert_file = root_ca_path_;
        req_ptr->client_cert_chain = device_cert_path_;
        req_ptr->client_ap_priv_key_slot = slot_cfg_;
        req_ptr->sdkType = https::OPENSSL_AP;
        for (auto& it : https_param.headers) {
            req_ptr->headers[it.first] = it.second;
        }

        std::string response_body;
        TspComm::TspResponse result_https;
        HTTPS_INFO << "RequestHttps begin Communication.";
        int32_t result_com = Communication(req_ptr, response_body);
        HTTPS_INFO << "RequestHttps end Communication .";

        if ((result_com == static_cast<int32_t>(HttpsResultCode::HttpsResult_Success)) && response_body.size() > 0) {
            HTTPS_INFO << "RequestHttps response:" << response_body;
            result_https.result_code = HttpsResultCode::HttpsResult_Success;
        } else {
            HTTPS_WARN << "RequestHttps response:"<<response_body <<" result_code:" << result_com;
            result_https.result_code = result_com;
        }
        result_https.response = response_body;
        result_https.content_type = " ";
        promise_https->set_value(result_https);
    };

    {
        std::lock_guard<std::mutex> lock(th_info->task_queue_mutex);
        th_info->task_queue.push_back(task);
    }

    return promise_https->get_future();
}


TspComm::TspResponse TspComm::ParseHttpBodyInjson(ReqType req_typt,std::string response_body) {
    TspComm::TspResponse result;
    // Parse http body in jason.
    std::unique_ptr<Json::CharReader> reader(Json::CharReaderBuilder().newCharReader());
    Json::Value root;
    Json::String errs;
    if (reader->parse(response_body.data(), response_body.data() + response_body.size(), &root, &errs)) {
        std::string msg,description;
        uint32_t code = 0;
        std::string data;

        if(req_typt ==ReqType::kReqHdUuid || req_typt ==ReqType::kReqUploadToken){
            if (root.isMember("description") && root["description"].isString()) {
                description = root["description"].asString();
            }
            if (root.isMember("code") && root["code"].isIntegral()) {
                code = root["code"].asUInt();
            }
            if ((code == 20000) && root.isMember("data") && root["data"].isString()) {
                std::string data = root["data"].asString();
                result.result_code = HttpsResultCode::HttpsResult_Success;
                result.response = data;
            }
        }else if(req_typt ==ReqType::kReqUploadRemoteConfig){
            if (root.isMember("msg") && root["msg"].isString()) {
                msg = root["msg"].asString();
            }
            if (root.isMember("code") && root["code"].isIntegral()) {
                code = root["code"].asUInt();
            }
            if ((code == 20000) && root.isMember("data") && root["data"].isObject()) {
                std::string remote_config = root["data"].toStyledString();
                result.result_code = HttpsResultCode::HttpsResult_Success;
                result.response = remote_config;
            } else {
                result.result_code = code;
            }
        } else {
            result.result_code = code;
        }
    } else {
        HTTPS_ERROR << "Failed to parse remote config response json.";
    }
    return result;
}

HttpsResultCode TspComm::Communication(std::shared_ptr<https::Request> request, std::string& response_body) {
    HTTPS_INFO << "Communication request->client_cert_chain: " << request->client_cert_chain;
    HTTPS_INFO << "Communication request->client_ap_priv_key_slot: " << request->client_ap_priv_key_slot;

    // Define response handler.
    std::shared_ptr<std::promise<int32_t>> promise = std::make_shared<std::promise<int32_t>>();
    auto response_handler = [promise, &response_body](int id, https::ResponsePtr response) {
        TspComm::TspResponse result_token;
        if((response->status_download == https::Status::complete) || (response->status_download == https::Status::fail)){
            HTTPS_INFO << "Response. code = " << response->code << ", content = " << response->content;
            if (response->code == 200) {
                response_body = response->content;
                result_token.response = response->content;
                result_token.result_code = https::HttpsResultCode::HttpsResult_Success;
                promise->set_value(HttpsResultCode::HttpsResult_Success);
            }
            else {
                result_token.result_code = response->code;
                promise->set_value(response->code);
            }
        }
    };
    std::future<int32_t> fut = promise->get_future();
    
    if (0 <= Https::Instance().HttpRequest(request, response_handler)) {
    } else {
        // TODO: Report fault.
        promise->set_value(HttpsResultCode::HttpsResult_OtherError);
    }
    int32_t result = fut.get();
    return static_cast<HttpsResultCode>(result);
}

std::shared_ptr<TspComm::ThreadInfo> TspComm::GetThreadInfo(ReqType type) {
    std::shared_ptr<TspComm::ThreadInfo> th_info;
    {
        std::lock_guard<std::mutex> lock(th_map_mutex_);
        if (th_map_.find(type) != th_map_.end()) {
            th_info = th_map_[type];
        }
    }

    if (!th_info) {
        th_info = std::make_shared<ThreadInfo>();
        th_info->th = std::thread([th_info, this]() {
            while (!stopped_) {

                Task task;
                {
                    std::lock_guard<std::mutex> lock(th_info->task_queue_mutex);
                    if (th_info->task_queue.begin() != th_info->task_queue.end()) {
                        task = th_info->task_queue.front();
                        th_info->task_queue.pop_front();
                    }
                }
                if (task) {
                    task();
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });

        {
            std::lock_guard<std::mutex> lock(th_map_mutex_);
            th_map_[type] = th_info;
        }
    }

    return th_info;
}

bool TspComm::GetPkiConfig() {
    auto cfgMgr = hozon::netaos::cfg::ConfigParam::Instance();
    int32_t res = cfgMgr->Init(3000);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        HTTPS_ERROR << "config module init error, res :" << res;
        return false;
    }
    // slot cfg
    res = cfgMgr->GetParam<std::string>("pki/slot", slot_cfg_);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        HTTPS_ERROR << "get config of slot error, res :" << res;
        return false;
    }

    // vin
    res = cfgMgr->GetParam<std::string>("pki/vin", vin_);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        HTTPS_ERROR << "get config of vin error, res :" << res;
        return false;
    }

    // pki status
    res = cfgMgr->GetParam<uint8_t>("pki/status", device_cert_state_);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        HTTPS_ERROR << "get config of pki status error, res :" << res;
        return false;
    }

    // root ca path
    res = cfgMgr->GetParam<std::string>("pki/root_ca_path", root_ca_path_);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        HTTPS_ERROR << "get config of root ca path error, res :" << res;
        return res;
    }

    // root ca path device cert path
    res = cfgMgr->GetParam<std::string>("pki/device_cert_path", device_cert_path_);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        HTTPS_ERROR << "get config of device cert path error, res :" << res;
        return false;
    }

    HTTPS_INFO << "pki configs. slot_cfg_: " << slot_cfg_
               << ", vin_: " << vin_ 
               << ", device_cert_state_: " << device_cert_state_
               << ", root_ca_path_: "  << root_ca_path_
               << ", device_cert_path_: " << device_cert_path_;
    // cfgMgr->DeInit();
    return true;
}

}  // namespace https
}  // namespace netaos
}  // namespace hozon