#include "adf-lite/include/executor.h"
#include "adf-lite/include/executor_impl.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

ClassLoader<Executor> g_class_loader;

Executor::Executor() :
    _pimpl(new ExecutorImpl()) {

}

Executor::~Executor() {

}

void Executor::RegistAlgProcessFunc(const std::string& trigger, Executor::AlgProcessFunc func) {
    _pimpl->RegistAlgProcessFunc(trigger, func);
}

void Executor::RegistAlgProcessWithProfilerFunc(const std::string& trigger, AlgProcessWithProfilerFunc func) {
    _pimpl->RegistAlgProcessWithProfilerFunc(trigger, func);
}

void Executor::RegistPauseTriggerCb(PauseTriggerCb func) {
    _pimpl->RegistPauseTriggerCb(func);
}

int32_t Executor::SendOutput(Bundle* output) {
    return _pimpl->SendOutput(output);
}

int32_t Executor::SendOutput(Bundle* output, const ProfileToken& token) {
    return _pimpl->SendOutput(output, token);
}
int32_t Executor::SendOutput(const std::string& name, BaseDataTypePtr data) {
    return _pimpl->SendOutput(name, data);
}

int32_t Executor::SendOutput(const std::string& name, BaseDataTypePtr data, const ProfileToken& token) {
    return _pimpl->SendOutput(name, data, token);
}

Executor::AlgProcessFunc Executor::GetProcessFunc(const std::string& trigger) {
    return _pimpl->GetProcessFunc(trigger);
}

Executor::AlgProcessWithProfilerFunc Executor::GetProcessWithProfilerFunc(const std::string& trigger) {
    return _pimpl->GetProcessWithProfilerFunc(trigger);
}

std::string Executor::GetConfigFilePath() {
    return _pimpl->GetConfigFilePath();
}

void Executor::SetConfigFilePath(const std::string& path) {
    _pimpl->SetConfigFilePath(path);
}

ExecutorConfig* Executor::GetConfig() {
    return _pimpl->GetConfig();
}

void Executor::SetConfig(ExecutorConfig* config) {
    return _pimpl->SetConfig(config);
}

int32_t Executor::PauseTrigger(const std::string& trigger) {
    return _pimpl->PauseTrigger(trigger);
}

int32_t Executor::ResumeTrigger(const std::string& trigger) {
    return _pimpl->ResumeTrigger(trigger);
}
int32_t Executor::PauseTrigger() {
    return _pimpl->PauseTrigger();
}

int32_t Executor::ResumeTrigger() {
    return _pimpl->ResumeTrigger();
}

void Executor::AlgPreRelease() {
}

}
}
}

