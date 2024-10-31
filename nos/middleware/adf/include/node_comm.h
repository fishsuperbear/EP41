#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "adf/include/data_types/common/types.h"
#include "adf/include/node_config.h"
#include "adf/include/node_proxy.h"
#include "adf/include/node_skeleton.h"
#include "fastdds/dds/topic/TopicDataType.hpp"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace adf {

using PubSubTypeBasePtr = std::shared_ptr<eprosima::fastdds::dds::TopicDataType>;
using TypeClassMap = std::unordered_map<std::string, PubSubTypeBasePtr>;

class NodeCommRecvInstance {
   public:
    NodeCommRecvInstance();
    ~NodeCommRecvInstance();
    int32_t Create(const NodeConfig::CommInstanceConfig& config);
    BaseDataTypePtr GetOneDataBlocking(uint32_t blocktime_ms);
    BaseDataTypePtr GetOneData(const uint32_t freshDataTime);
    std::vector<BaseDataTypePtr> GetNdata(uint32_t n);
    void PauseReceive();
    void ResumeReceive();

    std::shared_ptr<NodeProxyBase> _recv_base;
};

class NodeCommSendInstance {
   public:
    NodeCommSendInstance();
    ~NodeCommSendInstance();
    int32_t Create(const NodeConfig::CommInstanceConfig& config);
    int32_t SendOneData(BaseDataTypePtr data);

    std::shared_ptr<NodeSkeletonBase> _send_base;
};

class NodeCommCmMap {
   public:
    static NodeCommCmMap& GetInstance() {
        static NodeCommCmMap instance;
        return instance;
    }

    NodeCommCmMap();
    ~NodeCommCmMap();
    int Create(const std::string& name, PubSubTypeBasePtr _pub_sub_type_map);
    PubSubTypeBasePtr Get(const std::string& name);

   private:
    TypeClassMap _pub_sub_type_map;
    static std::recursive_mutex _mtx;
    std::recursive_mutex _map_mt;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
