#pragma once

#include <atomic>
#include "adf-lite/ds/ds_recv/ds_recv.h"
#include "adf-lite/include/executor.h"
#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class IdlDsRecv : public DsRecv {
   public:
    IdlDsRecv(const DSConfig::DataSource& config);
    virtual ~IdlDsRecv();
    virtual void Deinit() override;
    virtual void PauseReceive() override;
    virtual void ResumeReceive() override;

   private:
    std::unique_ptr<hozon::netaos::cm::Proxy> _proxy;
    void OnDataReceive(void);

    std::atomic<bool> _initialized;
};

}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon