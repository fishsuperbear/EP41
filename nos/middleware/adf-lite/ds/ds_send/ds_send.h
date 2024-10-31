#pragma once

#include "adf-lite/ds/ds_config.h"
#include "adf-lite/ds/ds_logger.h"
#include "adf-lite/include/base.h"
#include "adf-lite/include/reader.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
    
class DsSend {
public:
    DsSend(const DSConfig::DataSource& config) : _config(config) {}
    virtual ~DsSend() {}
    virtual void PreDeinit() = 0;
    virtual void Deinit() = 0;
    virtual void PauseSend() = 0;
    virtual void ResumeSend() = 0;

protected:
    Reader _reader;
    DSConfig::DataSource _config;
};

}
}
}