#pragma once

#include <signal.h>
#include <cstdint>
#include <functional>
#include <condition_variable>
#include <mutex>

namespace hozon {
namespace netaos {

class SigHandler {
public:
    static void Init() {
        signal(SIGTERM, &SigHandler::SignalHandler);
        signal(SIGINT, &SigHandler::SignalHandler);
        _term_signal = false;
    }

    static void SignalHandler(int32_t sig) {
        switch (sig) {
        case SIGTERM:
        case SIGINT:
        {
            std::unique_lock<std::mutex> lk(_term_mutex);
            _term_signal = true;
            _term_cv.notify_all();
            break;
        }

        default:
            break;
        }
    }
    
    static void NeedStopBlocking() {
        std::unique_lock<std::mutex> lk(_term_mutex);
        _term_cv.wait(lk, []() {
            return _term_signal == true;
        });
    }

    static bool NeedStop() {
        std::unique_lock<std::mutex> lk(_term_mutex);
        return _term_signal == true;
    }

private:
    static bool _term_signal;
    static std::condition_variable _term_cv;
    static std::mutex _term_mutex;
};

}
}