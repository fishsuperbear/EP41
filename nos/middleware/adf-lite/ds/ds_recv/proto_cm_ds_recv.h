#pragma once

#include <atomic>
#include "adf-lite/ds/ds_recv/ds_recv.h"
#include "adf-lite/include/executor.h"
#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
    
class ProtoCMDsRecv : public DsRecv {
public:
    ProtoCMDsRecv(const DSConfig::DataSource& config);
    virtual ~ProtoCMDsRecv();
    virtual void Deinit() override;
    virtual void PauseReceive() override;
    virtual void ResumeReceive() override;

private:
    std::atomic<bool> _initialized;
    hozon::netaos::cm::Proxy _proxy;
    void OnDataReceive(void);
};

}
}
}