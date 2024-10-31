#include "adf/include/node_comm.h"
#include "adf/include/internal_log.h"
#include "adf/include/node_config.h"
#include "idl/generated/avmPubSubTypes.h"
#include "idl/generated/common.h"

//skeleton
#include "adf/include/skeleton/node_skeleton_cm.h"
#include "adf/include/skeleton/node_skeleton_proto.h"

// proxy
#include "adf/include/proxy/node_proxy_camera.h"
#include "adf/include/proxy/node_proxy_cm.h"
#include "adf/include/proxy/node_proxy_h265_to_yuv.h"
#include "adf/include/proxy/node_proxy_idl.h"
#include "adf/include/proxy/node_proxy_idl_cuda.h"
#include "adf/include/proxy/node_proxy_nvs_cuda_desay.h"
#include "adf/include/proxy/node_proxy_proto.h"
#include "adf/include/proxy/node_proxy_proto_cuda.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeCommRecvInstance::NodeCommRecvInstance() {}

NodeCommRecvInstance::~NodeCommRecvInstance() {}

template <typename T>
std::shared_ptr<NodeProxyBase> CreateProxy(const NodeConfig::CommInstanceConfig& config) {
    return std::make_shared<T>(config);
}

template <typename T>
std::shared_ptr<NodeProxyBase> CreateCMProxy(const NodeConfig::CommInstanceConfig& config) {
    PubSubTypeBasePtr pub_sub_type_ptr = NodeCommCmMap::GetInstance().Get(config.name);
    if (nullptr == pub_sub_type_ptr) {
        return nullptr;
    }
    return std::make_shared<T>(config, pub_sub_type_ptr);
}

template <typename T>
std::shared_ptr<NodeProxyBase> CreateProtoProxy(const NodeConfig::CommInstanceConfig& config) {
    return std::make_shared<T>(config, std::make_shared<CmProtoBufPubSubType>());
}

template <typename T>
std::shared_ptr<NodeProxyBase> CreateIdlProxy(const NodeConfig::CommInstanceConfig& config) {
    return std::make_shared<T>(config);
}

std::unordered_map<std::string, std::function<std::shared_ptr<NodeProxyBase>(const NodeConfig::CommInstanceConfig&)>>
    g_proxy_type_map = {
        {"cm_proxy", CreateCMProxy<NodeProxyCM>},
        {"proto_proxy", CreateProtoProxy<NodeProxyProto>},
        {"idl_proxy", CreateIdlProxy<NodeProxyIdl>},
        {"idl_cuda_proxy", CreateIdlProxy<NodeProxyIdlCuda>},
        {"h265_to_yuv_proxy", CreateProtoProxy<NodeProxyH265ToYUV>},
        {"proto_cuda_proxy", CreateProtoProxy<NodeProxyProtoCuda>},
#ifdef BUILD_FOR_ORIN
        {"camera_proxy", CreateProxy<NodeProxyCamera>},
        {"nvs_cuda_proxy", CreateProxy<NodeProxyNVSCUDADesay>},
#endif
};

int32_t NodeCommRecvInstance::Create(const NodeConfig::CommInstanceConfig& config) {
    const auto& it = g_proxy_type_map.find(config.type);
    if (it == g_proxy_type_map.end()) {
        ADF_LOG_ERROR << "Fail to find" << config.type << " in proxy map, proxy map size " << g_proxy_type_map.size();
        return -1;
    }
    _recv_base = it->second(config);
    return 0;
}

BaseDataTypePtr NodeCommRecvInstance::GetOneDataBlocking(uint32_t blocktime_ms) {
    return _recv_base->GetOneDataBlocking(blocktime_ms);
}

BaseDataTypePtr NodeCommRecvInstance::GetOneData(const uint32_t freshDataTime) {
    return _recv_base->GetOneData(freshDataTime);
}

std::vector<BaseDataTypePtr> NodeCommRecvInstance::GetNdata(uint32_t n) {
    return _recv_base->GetNdata(n);
}

void NodeCommRecvInstance::PauseReceive() {
    _recv_base->PauseReceive();
}

void NodeCommRecvInstance::ResumeReceive() {
    _recv_base->ResumeReceive();
}

NodeCommSendInstance::NodeCommSendInstance() {}

NodeCommSendInstance::~NodeCommSendInstance() {}

template <typename T>
std::shared_ptr<NodeSkeletonBase> CreateSkeleton(const NodeConfig::CommInstanceConfig& config) {
    return std::make_shared<T>(config.domain, config.topic, config.buffer_capacity, config.is_async);
}

template <typename T>
std::shared_ptr<NodeSkeletonBase> CreateCMSkeleton(const NodeConfig::CommInstanceConfig& config) {
    PubSubTypeBasePtr pub_sub_type_ptr = NodeCommCmMap::GetInstance().Get(config.name);
    if (pub_sub_type_ptr == nullptr) {
        return nullptr;
    }
    return std::make_shared<T>(config, pub_sub_type_ptr);
}

template <typename T>
std::shared_ptr<NodeSkeletonBase> CraeteProtoSkeleton(const NodeConfig::CommInstanceConfig& config) {
    return std::make_shared<T>(config, std::make_shared<CmProtoBufPubSubType>());
}

std::unordered_map<std::string, std::function<std::shared_ptr<NodeSkeletonBase>(const NodeConfig::CommInstanceConfig&)>>
    g_skeleton_type_map = {
        {"cm_skeleton", CreateCMSkeleton<NodeSkeletonCM>},
        {"proto_skeleton", CraeteProtoSkeleton<NodeSkeletonProto>},
};

int32_t NodeCommSendInstance::Create(const NodeConfig::CommInstanceConfig& config) {
    const auto& it = g_skeleton_type_map.find(config.type);
    if (it == g_skeleton_type_map.end()) {
        ADF_LOG_ERROR << "Fail to find " << config.type << "in skeleton map, skeleton map size "
                      << g_skeleton_type_map.size();
        return -1;
    }
    _send_base = it->second(config);
    return 0;
}

int32_t NodeCommSendInstance::SendOneData(BaseDataTypePtr data) {
    return _send_base->SendOneData(data);
}

NodeCommCmMap::NodeCommCmMap() {}

NodeCommCmMap::~NodeCommCmMap() {}

int NodeCommCmMap::Create(const std::string& name, PubSubTypeBasePtr pub_sub_type) {
    std::lock_guard<std::recursive_mutex> lck(_map_mt);
    if (_pub_sub_type_map.find(name) != _pub_sub_type_map.end()) {
        ADF_EARLY_LOG << "CM type class " << name << " exist.";
        return -1;
    }
    _pub_sub_type_map[name] = pub_sub_type;
    return 0;
}

PubSubTypeBasePtr NodeCommCmMap::Get(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lck(_map_mt);
    const auto& it = _pub_sub_type_map.find(name);
    if (it == _pub_sub_type_map.end()) {
        ADF_LOG_ERROR << "Fail find " << name << ", cm type map size " << _pub_sub_type_map.size();
        return nullptr;
    }
    return it->second;
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
