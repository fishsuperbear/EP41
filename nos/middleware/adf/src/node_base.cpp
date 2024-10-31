#include "adf/include/node_base.h"
#include "adf/include/class_loader.h"
#include "adf/include/node_base_impl.h"
#include "adf/include/node_profiler_token.h"

namespace hozon {
namespace netaos {
namespace adf {
ClassLoader<hozon::netaos::adf::NodeBase> g_class_loader;

NodeBase::NodeBase() : _pimpl(new NodeBaseImpl(this)) {}

NodeBase::~NodeBase() {}

int32_t NodeBase::Start(const std::string& config_file, bool log_already_inited) {
    return _pimpl->Start(config_file, log_already_inited);
}

void NodeBase::Stop() {
    _pimpl->Stop();
}

bool NodeBase::NeedStop() {
    return _pimpl->NeedStop();
}

bool NodeBase::NeedStopBlocking() {
    return _pimpl->NeedStopBlocking();
}

void NodeBase::BypassSend() {
    _pimpl->BypassSend();
}

void NodeBase::BypassRecv() {
    _pimpl->BypassRecv();
}

void NodeBase::RegistAlgProcessFunc(const std::string& trigger, AlgProcessFunc func) {
    _pimpl->RegistAlgProcessFunc(trigger, func);
}

void NodeBase::RegistAlgProcessWithProfilerFunc(const std::string& trigger, AlgProcessWithProfilerFunc func) {
    _pimpl->RegistAlgProcessWithProfilerFunc(trigger, func);
}

std::shared_ptr<ThreadPool> NodeBase::GetThreadPool() {
    return _pimpl->GetThreadPool();
}

int32_t NodeBase::SendOutput(NodeBundle* output) {
    return _pimpl->SendOutput(output);
}

int32_t NodeBase::SendOutput(NodeBundle* output, const ProfileToken& token) {
    return _pimpl->SendOutput(output, token);
}

std::vector<std::string> NodeBase::GetTriggerList() {
    return _pimpl->GetTriggerList();
}

std::vector<std::string> NodeBase::GetAuxSourceList(const std::string& trigger_name) {
    return _pimpl->GetAuxSourceList(trigger_name);
}

const NodeConfig& NodeBase::GetConfig() {
    return _pimpl->GetConfig();
}

int32_t NodeBase::PauseTrigger(const std::string& trigger) {
    return _pimpl->PauseTrigger(trigger);
}

int32_t NodeBase::PauseTriggerAndJoin(const std::string& trigger) {
    return _pimpl->PauseTriggerAndJoin(trigger);
}

int32_t NodeBase::ResumeTrigger(const std::string& trigger) {
    return _pimpl->ResumeTrigger(trigger);
}

void NodeBase::RegisterCMType(const std::string& name,
                              std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type) {
    _pimpl->RegisterCMType(name, pub_sub_type);
}

int32_t NodeBase::InitLoggerStandAlone(const std::string& config_file) {
    return hozon::netaos::adf::NodeBaseImpl::InitLoggerStandAlone(config_file);
}

void NodeBase::ReportFault(uint32_t _faultId, uint8_t _faultObj) {
    _pimpl->ReportFault(_faultId, _faultObj);
}
}  // namespace adf
}  // namespace netaos
}  // namespace hozon