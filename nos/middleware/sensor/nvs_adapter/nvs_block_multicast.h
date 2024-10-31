#pragma once

#include "sensor/nvs_adapter/nvs_block_common.h"

namespace hozon {
namespace netaos {
namespace nv { 

class NVSBlockMulticast : public NVSBlockCommon {
public:
    int32_t Create(uint32_t num_consumer);

protected:
    // virtual void DeleteBlock();

};

}
}
}