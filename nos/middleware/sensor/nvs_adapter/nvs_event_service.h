#pragma once

#include "nvscievent.h"
#include "nvscistream.h"
#include "nvscievent.h"
#include <functional>
#include <vector>
#include <thread>
#include <memory>
#include "sensor/nvs_adapter/nvs_block_common.h"

namespace hozon {
namespace netaos {
namespace nv {

#define MAX_BLOCKS 100

class NVSEventService {
public:
    using OnEventCallback = std::function<int32_t(void)>;

    static NVSEventService& GetInstance() {
        static NVSEventService instance;

        return instance;
    }

    int32_t Init();
    int32_t Reg(NVSBlockCommon* nvs_block);
    int32_t Loop();
    void Run();
    void Deinit();

private:
    struct BlockEventData {
        NVSBlockCommon* nvs_block;
        NvSciEventNotifier* notifier;
        bool enable;
    };

    NvSciEventLoopService* _service = nullptr;
    std::shared_ptr<std::thread> _event_poll_thread;
    int32_t _num_blocks = 0;
    BlockEventData _blocks[MAX_BLOCKS];
    bool _running = true;

};

}    
}    
}