#include "sensor/nvs_adapter/nvs_block_queue.h"
#include "sensor/nvs_adapter/nvs_logger.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NVSBlockQueue::Create(bool use_mailbox) {
    name = use_mailbox ? "MAILBOX" : "FIFO";

    NvSciError err = use_mailbox
                   ? NvSciStreamMailboxQueueCreate(&block)
                   : NvSciStreamFifoQueueCreate(&block);

    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to create queue block, ret " << LogHexNvErr(err);
        return -1;
    }

    RegIntoEventService();

    return 0;
}

}
}
}