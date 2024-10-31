#pragma once

#include "adf/include/data_types/image/orin_image.h"
#include "adf/include/node_proto_register.h"
#include "adf/include/proxy/node_proxy_cm.h"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyProtoCuda : public NodeProxyCM {
   public:
    explicit NodeProxyProtoCuda(const NodeConfig::CommInstanceConfig& config,
                                std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type);
    ~NodeProxyProtoCuda();

    void OnDataReceive(void) override;

   protected:
    std::shared_ptr<NvsImageCUDA> CvtImage2Cuda(const std::shared_ptr<hozon::soc::Image>& pb_Image);

    cudaStream_t cuda_stream_;
    bool cuda_memory_init = false;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
