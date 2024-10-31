#include "cm/include/method_adapter.h"
#include "cm/include/method_adapter_impl.h"

namespace hozon {
namespace netaos {
namespace cm {

MethodClientAdapter::MethodClientAdapter(
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type) :
        _pimpl(new MethodClientAdapterImpl(req_topic_type, resp_topic_type)) {

}

MethodClientAdapter::~MethodClientAdapter() {}

int32_t MethodClientAdapter::Init(const uint32_t domain, const std::string& service_name) {
    return _pimpl->Init(domain, service_name);
}

int32_t MethodClientAdapter::Deinit() {
    return _pimpl->Deinit();
}

int32_t MethodClientAdapter::WaitServiceOnline(int64_t timeout_ms) {
    return _pimpl->WaitServiceOnline(timeout_ms);
}

int32_t MethodClientAdapter::Request(std::shared_ptr<ServiceBase> req, std::shared_ptr<ServiceBase> resp, int64_t timeout_ms) {
    return _pimpl->Request(req, resp, timeout_ms);
}

int32_t MethodClientAdapter::RequestAndForget(std::shared_ptr<ServiceBase> req) {
    return _pimpl->RequestAndForget(req);
}

void MethodClientAdapter::RegisterGenResp(GenServiceBaseFunc func) {
    return _pimpl->RegisterGenResp(func);
}

MethodServerAdapter::MethodServerAdapter(
    std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
    std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type) : 
        _pimpl(new MethodServerAdapterImpl(req_topic_type, resp_topic_type)) {

}

MethodServerAdapter::~MethodServerAdapter() {}

int32_t MethodServerAdapter::Start(const uint32_t& domain, const std::string& service_name) {
    return _pimpl->Start(domain, service_name);
}

int32_t MethodServerAdapter::Stop() {
    return _pimpl->Stop();
}

void MethodServerAdapter::RegisterProcess(ProcessFunc cb) {
    _pimpl->RegisterProcess(cb);
}

void MethodServerAdapter::RegisterGenReq(GenServiceBaseFunc func) {
    _pimpl->RegisterGenReq(func);
}

void MethodServerAdapter::RegisterGenResp(GenServiceBaseFunc func) {
    _pimpl->RegisterGenResp(func);
}

void MethodClientAdapter::RegisterAssignmentResp(GenAssignmentRespFunc func) 
{ 
    _pimpl->RegisterAssignmentResp(func); 
}

}  // namespace cm
}  // namespace netaos
}  // namespace hozon
