#pragma once

#include "sensor/nvs_adapter/nvs_block_common.h"

namespace hozon {
namespace netaos {
namespace nv { 

class NVSBlockIPCSrc : public NVSBlockCommon {
public:
    int32_t Create(const std::string& channel_name);

private:
    NvSciIpcEndpoint _ipc_endpoint = 0;
};

}
}
}