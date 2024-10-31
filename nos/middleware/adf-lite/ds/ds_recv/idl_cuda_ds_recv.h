#pragma once

#include <atomic>
#include "adf-lite/ds/ds_recv/ds_recv.h"
#include "adf-lite/include/data_types/image/orin_image.h"
#include "adf-lite/include/executor.h"
#include "cm/include/proxy.h"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class IdlCudaDsRecv : public DsRecv {
   public:
    IdlCudaDsRecv(const DSConfig::DataSource& config);
    virtual ~IdlCudaDsRecv();
    virtual void Deinit() override;
    virtual void PauseReceive() override;
    virtual void ResumeReceive() override;

   private:
    std::unique_ptr<hozon::netaos::cm::Proxy> _proxy;
    void OnDataReceive8M420(void);
    void OnDataReceive2M422(void);
    std::shared_ptr<NvsImageCUDA> CvtImage2Cuda(const std::shared_ptr<hozon::soc::Image>& pb_Image);

    cudaStream_t cuda_stream_;
    bool cuda_memory_init = false;
    std::atomic<bool> _initialized;
};

}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon