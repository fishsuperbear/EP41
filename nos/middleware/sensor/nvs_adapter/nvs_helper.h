#pragma once

#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvsciipc.h"
#include "sensor/nvs_adapter/nvs_logger.h"
#include <mutex>

namespace hozon {
namespace netaos {
namespace nv {

class NVSHelper {
public:
    static NVSHelper& GetInstance() {
        static NVSHelper instance;

        return instance;
    }

    int32_t Init() {
        std::call_once(_init_flag, [&](){ 
            NvSciError ret = NvSciError_Unknown;
            ret = NvSciSyncModuleOpen(&sci_sync_module);
            if (ret != NvSciError_Success) {
                NVS_LOG_CRITICAL << "Unable to open NvSciSync module, ret " << LogHexNvErr(ret);
                return;
            }

            ret = NvSciBufModuleOpen(&sci_buf_module);
            if (ret != NvSciError_Success) {
                NVS_LOG_CRITICAL << "Unable to open NvSciBuf module, ret " << LogHexNvErr(ret);
                return;
            }

            ret = NvSciIpcInit();
            if (ret != NvSciError_Success) {
                NVS_LOG_CRITICAL << "Unable to initialize NvSciIpc, ret " << LogHexNvErr(ret);
                return;
            }

            NVS_LOG_INFO << "Succ to init NvSciSync & NvSciBuf & NvSciIpc";
            return;
        });

        return 0;
    }

    NvSciSyncModule sci_sync_module;
    NvSciBufModule sci_buf_module;

private:
    NVSHelper() {}

    std::once_flag _init_flag;
};

}
}
}