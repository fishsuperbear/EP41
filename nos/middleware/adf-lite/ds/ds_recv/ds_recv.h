#pragma once

#include "adf-lite/ds/ds_config.h"
#include "adf-lite/ds/ds_logger.h"
#include "adf-lite/include/base.h"
#include "adf-lite/include/writer.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
    
class DsRecv {
public:
    DsRecv(const DSConfig::DataSource& config) : _config(config) {}
    virtual ~DsRecv() {}
    virtual void Deinit() = 0;
    virtual void PauseReceive() = 0;
    virtual void ResumeReceive() = 0;

protected:
    Writer _writer;
    DSConfig::DataSource _config;
};

}
}
}