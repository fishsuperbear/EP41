#pragma once

#ifdef BUILD_FOR_ORIN

#include "adf/include/internal_log.h"
#include "adf/include/node_proto_register.h"
#include "adf/include/node_proxy.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyCamera : public NodeProxyBase {
   public:
    explicit NodeProxyCamera(const NodeConfig::CommInstanceConfig& config);
    ~NodeProxyCamera();

    void OnDataReceive(void) override;
    void PauseReceive() override;
    void ResumeReceive() override;
    void Deinit() override;

   protected:
    BaseDataTypePtr CreateBaseDataFromProto(std::shared_ptr<google::protobuf::Message> msg);
    void ParseProtoHeader(std::shared_ptr<google::protobuf::Message> proto_msg, Header& header);
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon

#endif
