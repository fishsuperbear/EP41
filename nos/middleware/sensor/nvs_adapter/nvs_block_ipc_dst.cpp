#include "sensor/nvs_adapter/nvs_block_ipc_dst.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NVSBlockIPCDst::Create(const std::string& channel_name) {
    name = "IPC DST";

    NvSciError err;
    /* Open the named channel */
    err = NvSciIpcOpenEndpoint(channel_name.c_str(), &_ipc_endpoint);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to open channel " << channel_name << ", ret " << LogHexNvErr(err);
        return -1;
    }


    err = NvSciIpcResetEndpointSafe(_ipc_endpoint);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to reset ipc endpoint, ret " << LogHexNvErr(err);
        return -2;
    }

    /* Create a ipcdst block */
    err = NvSciStreamIpcDstCreate(_ipc_endpoint,
                                  NVSHelper::GetInstance().sci_sync_module,
                                  NVSHelper::GetInstance().sci_buf_module,
                                  &block);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to create ipc block, ret " << LogHexNvErr(err);
        return -3;
    }

    RegIntoEventService();

    return 0;
}

}
}
}