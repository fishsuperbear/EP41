#pragma once

#include <functional>
#include <future>
#include <memory>
#include <unordered_map>
#include <vector>
#include "adf/include/node_bundle.h"
#include "adf/include/node_config.h"
#include "adf/include/node_profiler_token.h"
#include "adf/include/node_proto_register.h"
#include "adf/include/thread_pool.h"
#include "fastdds/dds/topic/TopicDataType.hpp"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"

#define REGISTER_CM_TYPE_CLASS(name, pub_sub_type_class)                                               \
    {                                                                                                  \
        std::shared_ptr<pub_sub_type_class> pub_sub_type_obj = std::make_shared<pub_sub_type_class>(); \
        hozon::netaos::adf::NodeBase::RegisterCMType(name, pub_sub_type_obj);                          \
    }

#define REGISTER_CM_PROTOBUF_CLASS(name) REGISTER_CM_TYPE_CLASS(name, CmProtoBufPubSubType);

#define CM_PROTOBUF_SEND_DATA(send_name, protoValue, output)                     \
    {                                                                            \
        std::shared_ptr<CmProtoBuf> proto_data = std::make_shared<CmProtoBuf>(); \
        proto_data->name(protoValue.GetTypeName());                              \
        std::string str_data;                                                    \
        protoValue.SerializeToString(&str_data);                                 \
        proto_data->str(str_data);                                               \
        std::shared_ptr<hozon::netaos::adf::AlgDataBase> alg_data =              \
            std::make_shared<hozon::netaos::adf::AlgDataBase>();                 \
        alg_data->data = proto_data;                                             \
        output.Add(send_name, alg_data);                                         \
    }

#define CM_PROTOBUF_RECV_DATA(adf_recv_name, proto_recv_name, proto_recv_str)                              \
    {                                                                                                      \
        std::shared_ptr<hozon::netaos::adf::AlgDataBase> alg_data = input->GetOne(adf_recv_name);          \
        if (alg_data == nullptr) {                                                                         \
            NODE_LOG_ERROR << "Fail to get CM protobuf data";                                              \
            proto_recv_name = nullptr;                                                                     \
            proto_recv_str = nullptr;                                                                      \
        } else {                                                                                           \
            std::shared_ptr<CmProtoBuf> proto_data = std::static_pointer_cast<CmProtoBuf>(alg_data->data); \
            proto_recv_name = proto_data->name();                                                          \
            proto_recv_str = proto_data->str();                                                            \
        }                                                                                                  \
    }

namespace hozon {
namespace netaos {
namespace adf {
class NodeBaseImpl;

class NodeBase {
   public:
    NodeBase();
    virtual ~NodeBase();
    int32_t Start(const std::string& config_file, bool log_already_inited = false);
    void Stop();
    bool NeedStop();
    bool NeedStopBlocking();

    virtual int32_t AlgInit() = 0;
    virtual void AlgRelease() = 0;

    using AlgProcessFunc = std::function<int32_t(NodeBundle* input)>;
    void RegistAlgProcessFunc(const std::string& trigger, AlgProcessFunc func);

    using AlgProcessWithProfilerFunc = std::function<int32_t(NodeBundle* input, const ProfileToken& token)>;
    void RegistAlgProcessWithProfilerFunc(const std::string& trigger, AlgProcessWithProfilerFunc func);

    std::shared_ptr<ThreadPool> GetThreadPool();

    int32_t SendOutput(NodeBundle* output);
    int32_t SendOutput(NodeBundle* output, const ProfileToken& token);

    const NodeConfig& GetConfig();

    // just for internal use
    void BypassSend();
    void BypassRecv();

    int32_t PauseTrigger(const std::string& trigger);
    // DO NOT call PauseTriggerAndJoin within corresponding process function
    int32_t PauseTriggerAndJoin(const std::string& trigger);
    int32_t ResumeTrigger(const std::string& trigger);
    std::vector<std::string> GetTriggerList();
    std::vector<std::string> GetAuxSourceList(const std::string& trigger_name);
    void RegisterCMType(const std::string& name, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type);

    static int32_t InitLoggerStandAlone(const std::string& config_file);

    void ReportFault(uint32_t _faultId, uint8_t _faultObj);

   private:
    std::unique_ptr<NodeBaseImpl> _pimpl;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon
