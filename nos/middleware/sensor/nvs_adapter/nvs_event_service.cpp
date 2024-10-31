#include "sensor/nvs_adapter/nvs_event_service.h"
#include "sensor/nvs_adapter/nvs_logger.h"

namespace hozon {
namespace netaos {
namespace nv {

int32_t NVSEventService::Init() {
    void* os_config = nullptr;

    NvSciError err = NvSciEventLoopServiceCreateSafe(1U, os_config, &_service);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Failed to create event service, ret " << LogHexNvErr(err);
        return -1;
    }

    NVS_LOG_INFO << "Succ to create event service";
    return 0;
}

int32_t NVSEventService::Reg(NVSBlockCommon* nvs_block) {
    BlockEventData& event = _blocks[_num_blocks++];
    
    event.enable = true;
    event.nvs_block = nvs_block;
    NvSciError err = NvSciStreamBlockEventServiceSetup(event.nvs_block->block,
                                          &_service->EventService,
                                          &event.notifier);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Failed to create event notifier for block, ret " << LogHexNvErr(err);
        return -1;
    }

    NVS_LOG_INFO << "Succ to add event.";
    return 0;
}

int32_t NVSEventService::Loop() {
    NvSciEventNotifier* notifiers[MAX_BLOCKS];
    bool event[MAX_BLOCKS];

    for (int32_t i = 0; i < _num_blocks; ++i) {
        notifiers[i] = _blocks[i].notifier;
        event[i] = false;
    }

    NVS_LOG_INFO << "Enter event service loop, event count " << _num_blocks;
    while (_running) {
        memset(event, 0, sizeof(event));
        // NVS_LOG_DEBUG << "\nWaitForMultipleEventsExt timeout " << timeout;
        NvSciError err = _service->WaitForMultipleEventsExt(
                                                        &_service->EventService,
                                                        notifiers,
                                                        _num_blocks,
                                                        1000,
                                                        event);
        if ((NvSciError_Success != err) && (NvSciError_Timeout != err)) {
            NVS_LOG_CRITICAL << "Failed to wait/poll event service, ret " << LogHexNvErr(err);
            return -1;
        }

        /*
         * Check for events on new blocks that signaled or old blocks that
         *   had an event on the previous pass. This is done in reverse
         *   of the order in which blocks were registered. This is because
         *   producers are created before consumers, and for mailbox mode
         *   we want to give the consumer a chance to use payloads before
         *   the producer replaces them.
         */
        for (int32_t i = _num_blocks - 1; i >= 0; --i) {
            BlockEventData* block = &_blocks[i];
            if (block->enable && event[i]) {
                /* Call the block's event handler function */
                int32_t rv = block->nvs_block->EventHandler();
                if (rv < 0) {
                    block->enable = false;
                    NVS_LOG_INFO << "Disable block " << block->nvs_block->name;
                } 
            }
        }
    }

    /* Delete notifiers */
    for (int32_t i = 0; i < _num_blocks; ++i) {
        notifiers[i]->Delete(notifiers[i]);
    }

    /* Delete service */
    _service->EventService.Delete(&_service->EventService);

    return 0;
}

void NVSEventService::Run() {
    _running = true;
    _event_poll_thread = std::make_shared<std::thread>(&NVSEventService::Loop, this);
}

void NVSEventService::Deinit() {
    _running = false;
    _event_poll_thread->join();
}

}    
}    
}