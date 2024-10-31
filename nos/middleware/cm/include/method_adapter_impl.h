#pragma once

#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <string>

#include <uuid/uuid.h>

#include "cm/include/method_logger.h"
#include "cm/include/proxy.h"
#include "cm/include/skeleton.h"
#include "idl/generated/servicebase.h"
#include "cm/include/method_adapter.h"

namespace hozon {
namespace netaos {
namespace cm {

class MethodClientAdapterImpl {
public:
    explicit MethodClientAdapterImpl(
            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resq_topic_type);
            
    ~MethodClientAdapterImpl()=default;
    
    int32_t Init(const uint32_t domain, const std::string& service_name);
    int32_t Deinit();
    int32_t WaitServiceOnline(int64_t timeout_ms);
    int32_t Request(const std::shared_ptr<ServiceBase> &req, std::shared_ptr<ServiceBase> resp, int64_t timeout_ms);
    int32_t RequestAndForget(const std::shared_ptr<ServiceBase> &req);

    void RegisterGenResp(MethodClientAdapter::GenServiceBaseFunc func);
    void RegisterAssignmentResp(MethodClientAdapter::GenAssignmentRespFunc func);
    bool WaitRequest(std::string tmp_id, std::shared_ptr<ServiceBase>& resp, int64_t timeout_ms);

private:
    Skeleton skeleton_;
    Proxy proxy_;
    std::string request_service_name_;
    std::string response_service_name_;
    uint32_t domain_;
    std::array<char, 50> client_id_;
    // std::atomic<uint32_t> seq_;
    uint32_t seq_;
    MethodClientAdapter::GenServiceBaseFunc gen_resp_func_;
    std::atomic<bool> stop_flag_;  //为true时表示停止，为false时表示运行，程序初始化时候为false，析构时为true
    std::thread take_thread_;
    std::map<std::string, std::shared_ptr<ServiceBase>> resp_map_buffer_;
    std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type_;
    std::mutex buffer_resp_mutex_;
    std::condition_variable cv_;
    MethodClientAdapter::GenAssignmentRespFunc assignment_resp_;
    std::mutex data_mutex_;

};

class MethodServerAdapterImpl {
public:
    MethodServerAdapterImpl(
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type);
    ~MethodServerAdapterImpl();
    int32_t Start(const uint32_t& domain, const std::string& service_name);
    int32_t Stop();

    void RegisterProcess(MethodServerAdapter::ProcessFunc func);

    void RegisterGenReq(MethodServerAdapter::GenServiceBaseFunc func);
    void RegisterGenResp(MethodServerAdapter::GenServiceBaseFunc func);

private:
    bool serving_;
    Skeleton skeleton_;     
    Proxy proxy_;
    std::string request_service_name_;
    std::string response_service_name_;
    uint32_t domain_;
    MethodServerAdapter::ProcessFunc process_;
    MethodServerAdapter::GenServiceBaseFunc gen_req_func_;
    MethodServerAdapter::GenServiceBaseFunc gen_resp_func_;
};

}  // namespace cm
}  // namespace netaos
}  // namespace hozon
