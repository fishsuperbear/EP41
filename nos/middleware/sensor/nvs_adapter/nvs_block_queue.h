#pragma once

#include "sensor/nvs_adapter/nvs_block_common.h"
#include <string>

namespace hozon {
namespace netaos {
namespace nv { 

class NVSBlockQueue : public NVSBlockCommon {
public:
    int32_t Create(bool use_mailbox);
};

}
}
}