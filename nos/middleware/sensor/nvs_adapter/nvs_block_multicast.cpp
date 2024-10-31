#include "sensor/nvs_adapter/nvs_block_multicast.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NVSBlockMulticast::Create(uint32_t num_consumer) {
    name = "MULTICAST";

    NvSciError err = NvSciStreamMulticastCreate(num_consumer, &block);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to create multicast, ret " << LogHexNvErr(err); 
        return -1;
    }

    RegIntoEventService();

    return 0;
}

}
}
}
