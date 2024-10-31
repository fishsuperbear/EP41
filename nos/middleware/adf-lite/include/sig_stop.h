#pragma once

#include <signal.h>
#include <cstdint>
#include <functional>
#include <condition_variable>
#include <mutex>
#include "adf-lite/include/adf_lite_internal_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

void SigHandleFunc(int32_t sig);

class SigHandler {
public:
    static SigHandler& GetInstance() {
        static SigHandler instance;

        return instance;
    }

    void Init() {
        signal(SIGTERM, SigHandleFunc);
        signal(SIGINT, SigHandleFunc);
        _term_signal = false;
    }
    
    void NeedStopBlocking() {
        std::unique_lock<std::mutex> lk(_term_mutex);
        _term_cv.wait(lk, [this]() {
            return _term_signal == true;
        });
    }

    bool NeedStop() {
        std::unique_lock<std::mutex> lk(_term_mutex);
        return _term_signal == true;
    }

    bool _term_signal;
    std::condition_variable _term_cv;
    std::mutex _term_mutex;
};

void SigHandleFunc(int32_t sig) {
    switch (sig) {
    case SIGTERM:
    case SIGINT:
    {
        ADF_EARLY_LOG << "Receive Stop Signal sig= " << sig;
        std::unique_lock<std::mutex> lk(SigHandler::GetInstance()._term_mutex);
        SigHandler::GetInstance()._term_signal = true;
        SigHandler::GetInstance()._term_cv.notify_all();
        break;
    }

    default:
        break;
    }
}

}
}
}