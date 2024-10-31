#pragma once

#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <thread>
#include <list>
#include <iostream>
#include <cstdlib>
#include "adf-lite/include/bundle.h"
#include "adf-lite/include/classloader.h"
#include "adf-lite/include/adf_lite_logger.h"
#include "adf/include/node_profiler_token.h"
#define ALLTRIGGER_NAME "__AllTrigger__"
using namespace hozon::netaos::adf;
namespace hozon {
namespace netaos {
namespace adf_lite {

class ExecutorConfig;
class ExecutorImpl;

class Executor {
public:
    Executor();
    virtual ~Executor();

    virtual int32_t AlgInit() = 0;
    virtual void AlgRelease() = 0;
    virtual void AlgPreRelease();
    using AlgProcessFunc = std::function<int32_t(Bundle* input)>;
    void RegistAlgProcessFunc(const std::string& trigger, AlgProcessFunc func);
    using AlgProcessWithProfilerFunc = std::function<int32_t(Bundle* input, const ProfileToken& token)>;
    void RegistAlgProcessWithProfilerFunc(const std::string& trigger, AlgProcessWithProfilerFunc func);
    using PauseTriggerCb = std::function<int32_t(const std::string&, const bool)>;
    void RegistPauseTriggerCb(PauseTriggerCb func);

    int32_t SendOutput(Bundle* output);
    int32_t SendOutput(Bundle* output, const ProfileToken& token);
    int32_t SendOutput(const std::string& topic, BaseDataTypePtr data);
    int32_t SendOutput(const std::string& topic, BaseDataTypePtr data, const ProfileToken& token);
    AlgProcessFunc GetProcessFunc(const std::string& trigger);
    AlgProcessWithProfilerFunc GetProcessWithProfilerFunc(const std::string& trigger);
    std::string GetConfigFilePath();
    void SetConfigFilePath(const std::string& path);
    ExecutorConfig* GetConfig();
    void SetConfig(ExecutorConfig* config);
    int32_t PauseTrigger(const std::string& trigger);
    int32_t ResumeTrigger(const std::string& trigger);
    int32_t PauseTrigger();
    int32_t ResumeTrigger();
private:
    std::unique_ptr<ExecutorImpl> _pimpl;
    CtxLogger _exec_logger;
};

extern ClassLoader<Executor> g_class_loader;

#define REGISTER_DERIVED_CLASS(name, type, uniqueid) \
    class TypeInstance##name##uniqueid { \
    public: \
        TypeInstance##name##uniqueid() { \
            hozon::netaos::adf_lite::g_class_loader.RegisterClass<hozon::netaos::adf_lite::Executor>(#name, []() {return std::static_pointer_cast<hozon::netaos::adf_lite::Executor>(std::make_shared<type>()); }); \
        } \
    }; \
    static TypeInstance##name##uniqueid g_type_instance_##name##uniqueid;

#define REGISTER_ADF_CLASS_INTERNAL_1(name, type, uniqueid) \
    REGISTER_DERIVED_CLASS(name, type, uniqueid)


#define REGISTER_ADF_CLASS(name, type) \
    REGISTER_ADF_CLASS_INTERNAL_1(name, type, __COUNTER__)

#define AELOG_FATAL          CTX_LOG_FATAL(_exec_logger)
#define AELOG_ERROR          CTX_LOG_ERROR(_exec_logger)
#define AELOG_WARN           CTX_LOG_WARN(_exec_logger)
#define AELOG_INFO           CTX_LOG_INFO(_exec_logger)
#define AELOG_DEBUG          CTX_LOG_DEBUG(_exec_logger)
#define AELOG_VERBOSE        CTX_LOG_VERBOSE(_exec_logger)

}
}
}