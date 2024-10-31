#pragma once

#include <string>
#include <cstdint>
#include <future>

#include "cm/include/proxy.h"
#include "cm/include/skeleton.h"
#include "idl/generated/servicebase.h"

namespace hozon {
namespace netaos {
namespace cm {

class MethodClientAdapterImpl;
class MethodClientAdapter {
public:
    explicit MethodClientAdapter(
            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type);
    ~MethodClientAdapter();

    int32_t Init(const uint32_t domain, const std::string& service_name);
    int32_t Deinit();
    int32_t WaitServiceOnline(int64_t timeout_ms);
    int32_t Request(std::shared_ptr<ServiceBase> req, std::shared_ptr<ServiceBase> resp, int64_t timeout_ms);
    int32_t RequestAndForget(std::shared_ptr<ServiceBase> req);

    using GenServiceBaseFunc = std::function<std::shared_ptr<ServiceBase>()>;
    void RegisterGenResp(GenServiceBaseFunc func);

    using GenAssignmentRespFunc = std::function<void(std::shared_ptr<ServiceBase>&, std::shared_ptr<ServiceBase>&)>;
    void RegisterAssignmentResp(GenAssignmentRespFunc func);

private:
    std::unique_ptr<MethodClientAdapterImpl> _pimpl;
};

class MethodServerAdapterImpl;
class MethodServerAdapter {
public:
    MethodServerAdapter(
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type);
    ~MethodServerAdapter();

    int32_t Start(const uint32_t& domain, const std::string& service_name);
    int32_t Stop();

    using ProcessFunc = std::function<int32_t(const std::shared_ptr<ServiceBase>, std::shared_ptr<ServiceBase> resp)>;
    void RegisterProcess(ProcessFunc call);

    using GenServiceBaseFunc = std::function<std::shared_ptr<ServiceBase>()>;
    void RegisterGenReq(GenServiceBaseFunc func);
    void RegisterGenResp(GenServiceBaseFunc func);

private:
    std::unique_ptr<MethodServerAdapterImpl> _pimpl;
};
}  // namespace cm
}  // namespace netaos
}  // namespace hozon
