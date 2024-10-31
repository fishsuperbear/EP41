#include "cm/include/method_adapter_impl.h"
#include <memory>
#include <utility>

namespace hozon {
namespace netaos {
namespace cm {

MethodClientAdapterImpl::MethodClientAdapterImpl(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type)
    : skeleton_(req_topic_type), proxy_(resp_topic_type, 0x01), seq_(0), stop_flag_(false) {
    resp_topic_type_ = resp_topic_type;
};

int32_t MethodClientAdapterImpl::Init(const uint32_t domain, const std::string& service_name) {
    stop_flag_ = false;
    seq_ = 0;
    this->domain_ = domain;
    this->request_service_name_ = "/request/" + service_name;
    this->response_service_name_ = "/reply/" + service_name;
    uuid_t uuid;
    char temp_array[50] = {0};
    uuid_generate(uuid);
    uuid_unparse(uuid, temp_array);
    std::copy(std::begin(temp_array), std::end(temp_array), client_id_.begin());

    MD_LOG_INFO << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                << "],Topic:" << this->request_service_name_.substr(9) << ",Init start";

    if (skeleton_.Init(this->domain_, this->request_service_name_) < 0) {
        MD_LOG_ERROR << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                     << "],Topic:" << this->request_service_name_.substr(9)
                     << ",Skeleton init failed,response_skeleton_init: "
                     << skeleton_.Init(this->domain_, this->request_service_name_);
        return -1;
    }

    if (proxy_.Init(this->domain_, this->response_service_name_) < 0) {
        MD_LOG_ERROR << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                     << "],Topic:" << this->request_service_name_.substr(9) << ",Proxy init failed,request_proxy_init: "
                     << proxy_.Init(this->domain_, this->response_service_name_);
        return -1;
    }

    MD_LOG_INFO << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                << "],Topic:" << this->request_service_name_.substr(9) << ",Init end";

    proxy_.Listen([this](){
            std::shared_ptr<ServiceBase> resp(static_cast<ServiceBase*>(resp_topic_type_->createData()));
            if (0 == proxy_.Take(resp)) {
                std::unique_lock<std::mutex> lock(buffer_resp_mutex_);
                if (static_cast<std::string>(client_id_.data()) == static_cast<std::string>(resp->sender().data())) {
                    resp_map_buffer_[resp->sender().data() + std::to_string(resp->seq())] = resp;
                    MD_LOG_TRACE << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                                << "],Topic:" << this->request_service_name_.substr(9)
                                << ",Receive response:" << std::to_string(resp->seq());
                    cv_.notify_all();
                } else {
                    MD_LOG_TRACE << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                                << "],Topic:" << this->request_service_name_.substr(9)
                                << ",Receive other response:" << std::to_string(resp->seq());
                }
            } else {
                MD_LOG_ERROR << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                             << "],Topic:" << this->request_service_name_.substr(9) << ",Take error";
            }
        }
    );

    return 0;
}

int32_t MethodClientAdapterImpl::Deinit() {
    MD_LOG_INFO << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                << "],Topic:" << this->request_service_name_.substr(9) << ",Deinit start";

    stop_flag_ = true;
    cv_.notify_all();

    {
        std::unique_lock<std::mutex> lock(buffer_resp_mutex_);
        resp_map_buffer_.empty();
    }

    skeleton_.Deinit();
    proxy_.Deinit();

    MD_LOG_INFO << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                << "],Topic:" << this->request_service_name_.substr(9) << ",Deinit end";
    return 0;
}

int32_t MethodClientAdapterImpl::WaitServiceOnline(int64_t timeout_ms) {
    while (timeout_ms >= 0) {
        if (!skeleton_.IsMatched() || !proxy_.IsMatched()) {
            // MD_LOG_TRACE << this->request_service_name_ << " service find successful,skeleton_is_matched: " << skeleton_.IsMatched();
            // MD_LOG_TRACE << this->response_service_name_ << " service find successful,proxy_is_matched: " << proxy_.IsMatched();
        } else {
            MD_LOG_TRACE << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                         << "],Success to find service:" << this->request_service_name_.substr(9);
            return 0;
        }
        if (timeout_ms > 10) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            timeout_ms -= 10;
        } else {
            MD_LOG_WARN << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                        << "],Failed to find service:" << this->request_service_name_.substr(9) << ", Because timeout";
            return -1;
        }
    }
    return -1;
}

bool MethodClientAdapterImpl::WaitRequest(std::string tmp_id, std::shared_ptr<ServiceBase>& resp, int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(buffer_resp_mutex_);

    //not find id,then wait for
    cv_.wait_for(lock, std::chrono::duration<double, std::milli>(timeout_ms), [this, &tmp_id]() {
        if (true == stop_flag_) {
            return true;
        }
        return resp_map_buffer_.find(tmp_id) != resp_map_buffer_.end();
    });

    if (resp_map_buffer_.find(tmp_id) != resp_map_buffer_.end()) {
        assignment_resp_(resp, resp_map_buffer_[tmp_id]);
        MD_LOG_DEBUG << "Client[" << tmp_id.substr(0, 4) << "],Topic:" << this->request_service_name_.substr(9)
                     << ",Success to find response:" << resp->seq();
        resp_map_buffer_.erase(tmp_id);
        return true;
    } else {
        MD_LOG_DEBUG << "Client[" << tmp_id.substr(0, 4) << "],Topic:" << this->request_service_name_.substr(9)
                     << ",Failed to find response:" << resp->seq();
        return false;
    }
}

int32_t MethodClientAdapterImpl::Request(const std::shared_ptr<ServiceBase>& req, std::shared_ptr<ServiceBase> resp,
                                         int64_t timeout_ms) {
    if (timeout_ms <= 0) {
        MD_LOG_ERROR << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4) << "],Topic:"
                     << this->request_service_name_.substr(9) << ",Timeout_ms not legal,timeout_ms: " << timeout_ms;
        return -1;
    }

    //服务端没上线的case
    if (0 != WaitServiceOnline(0)) {
        return -1;
    }

    uint32_t seq_tmp;
    {
        std::lock_guard<std::mutex> write_lock(data_mutex_);
        seq_++;
        req->sender(client_id_);
        req->seq(seq_);
        seq_tmp = seq_;
        req->fire_forget(false);
        req->reply(0);  //有疑问

        if (skeleton_.Write(std::static_pointer_cast<void>(req)) != 0) {
            MD_LOG_ERROR << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4) << "],Topic:"
                         << this->request_service_name_.substr(9) << ",Failed to write request:" << req->seq();
            return -1;
        } else {
            MD_LOG_DEBUG << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4) << "],Topic:"
                         << this->request_service_name_.substr(9) << ",Success to write request:" << req->seq();
        }
    }

    while ((timeout_ms > 0) && (!stop_flag_)) {
        auto start = std::chrono::steady_clock::now();
        if (!WaitRequest(req->sender().data() + std::to_string(seq_tmp), resp, timeout_ms)) {
        } else {
            // 正常退出的地方
            MD_LOG_TRACE << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4)
                         << "],Topic:" << this->request_service_name_.substr(9) << ",Request:" << seq_tmp
                         << ",Successful exit" << " timout : " << timeout_ms;
            break;
        }

        auto end = std::chrono::steady_clock::now();
        timeout_ms -= (std::chrono::duration<double, std::milli>(end - start)).count();
    }

    if (timeout_ms <= 0 || stop_flag_) {
        if (timeout_ms <= 0) {
            MD_LOG_ERROR << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4) << "],Topic:"
                         << this->request_service_name_.substr(9)
                         << ",Failed to get response:" << std::to_string(seq_tmp) << ", Because timeout!" << timeout_ms;
        }
        if (stop_flag_) {
            MD_LOG_INFO << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4) << "],Topic:"
                        << this->request_service_name_.substr(9) << ",Received stop_signal in request processing";
        }
        return -1;
    }

    return resp->reply();
}

int32_t MethodClientAdapterImpl::RequestAndForget(const std::shared_ptr<ServiceBase>& req) {
    //服务端没上线的case
    if (0 != WaitServiceOnline(50)) {
        return -1;
    }

    std::lock_guard<std::mutex> write_lock(data_mutex_);
    seq_++;
    req->sender(client_id_);
    req->seq(seq_);
    req->fire_forget(true);
    req->reply(0);  //有疑问

    if (skeleton_.Write(std::static_pointer_cast<void>(req)) != 0) {
        MD_LOG_ERROR << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4) << "],Topic:"
                     << this->request_service_name_.substr(9) << ",Failed to write request(fire):" << req->seq();
        return -1;
    } else {
        MD_LOG_DEBUG << "Client[" << static_cast<std::string>(client_id_.data()).substr(0, 4) << "],Topic:"
                     << this->request_service_name_.substr(9) << ",Success to write request(fire):" << req->seq();
    }

    return 0;
}

void MethodClientAdapterImpl::RegisterGenResp(MethodClientAdapter::GenServiceBaseFunc func) {
    gen_resp_func_ = std::move(func);
}

void MethodClientAdapterImpl::RegisterAssignmentResp(MethodClientAdapter::GenAssignmentRespFunc func) {
    assignment_resp_ = std::move(func);
}

MethodServerAdapterImpl::MethodServerAdapterImpl(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type)
    : serving_(false), skeleton_(std::move(resp_topic_type)), proxy_(std::move(req_topic_type), 0x01), domain_(0) {}

MethodServerAdapterImpl::~MethodServerAdapterImpl() {}

int32_t MethodServerAdapterImpl::Start(const uint32_t& domain, const std::string& service_name) {
    serving_ = true;
    this->domain_ = domain;
    this->request_service_name_ = "/reply/" + service_name;
    this->response_service_name_ = "/request/" + service_name;

    if (skeleton_.Init(this->domain_, this->request_service_name_) < 0) {
        MD_LOG_ERROR << "Server[" << service_name << "],Failed to init skeleton";
        return -1;
    }

    if (proxy_.Init(this->domain_, this->response_service_name_) < 0) {
        MD_LOG_ERROR << "Server[" << service_name << "],Failed to init proxy";
        return -1;
    }

    MD_LOG_INFO << "Server[" << service_name << "],Init successful";
    proxy_.Listen([this, service_name]() {
        std::shared_ptr<ServiceBase> req = gen_req_func_();
        std::shared_ptr<ServiceBase> resp = gen_resp_func_();

        if (proxy_.Take(std::static_pointer_cast<void>(req)) != 0) {
            MD_LOG_ERROR << "Server[" << service_name << "],Failed to take Client["
                         << static_cast<std::string>(req->sender().data()).substr(0, 4) << "],Request:" << req->seq();
        } else {
            MD_LOG_DEBUG << "Server[" << service_name << "],Success to take Client["
                         << static_cast<std::string>(req->sender().data()).substr(0, 4) << "],Request:" << req->seq();
        }

        int32_t process_response = -1;

        if ((req != nullptr) && (resp != nullptr) && (process_ != nullptr)) {
            process_response = process_(req, resp);
            MD_LOG_TRACE << "Server[" << service_name << "],Client["
                        << static_cast<std::string>(req->sender().data()).substr(0, 4) << "],Request:" << req->seq()
                        << ",Call back successful";
        } else {
            MD_LOG_ERROR << "Server[" << service_name << "],Client["
                         << static_cast<std::string>(req->sender().data()).substr(0, 4) << "],Request:" << req->seq()
                         << ",Call back error!";
        }

        if (req->fire_forget()) {
        } else {
            resp->sender(req->sender());
            resp->seq(req->seq());
            resp->reply(process_response);

            if (0 != skeleton_.Write(std::static_pointer_cast<void>(resp))) {
                MD_LOG_ERROR << "Server[" << service_name << "],Failed to write Client["
                             << static_cast<std::string>(req->sender().data()).substr(0, 4)
                             << "],Response:" << req->seq();

            } else {
                MD_LOG_DEBUG << "Server[" << service_name << "],Success to write Client["
                             << static_cast<std::string>(req->sender().data()).substr(0, 4)
                             << "],Response:" << req->seq();
            }
        }
    });

    return 0;
}

int32_t MethodServerAdapterImpl::Stop() {
    serving_ = false;
    skeleton_.Deinit();
    proxy_.Deinit();
    MD_LOG_INFO << "Server[" << this->request_service_name_.substr(7) << "],Deinit successful";
    return 0;
}

void MethodServerAdapterImpl::RegisterProcess(MethodServerAdapter::ProcessFunc func) {
    process_ = std::move(func);
}

void MethodServerAdapterImpl::RegisterGenReq(MethodServerAdapter::GenServiceBaseFunc func) {
    gen_req_func_ = std::move(func);
}

void MethodServerAdapterImpl::RegisterGenResp(MethodServerAdapter::GenServiceBaseFunc func) {
    gen_resp_func_ = std::move(func);
}

}  // namespace cm
}  // namespace netaos
}  // namespace hozon
