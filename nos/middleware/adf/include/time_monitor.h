#include <unistd.h>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace hozon {
namespace netaos {
namespace adf {

class TimeMonitor {
   public:
    using CallBack = std::function<void(uint64_t)>;

    void Init(uint32_t period_ms, CallBack cb) {
        _period_ms = period_ms;
        _cb = cb;
        _routine = std::make_unique<std::thread>(&TimeMonitor::CheckRoutine, this);
    }

    void DeInit() {
        {
            std::unique_lock<std::mutex> lk(_mtx);
            _running = false;
            _begin_cv.notify_all();
            _end_cv.notify_all();
        }

        if (_routine) {
            _routine->join();
        }
    }

    void FeedBegin() {
        std::unique_lock<std::mutex> lk(_mtx);

        _begin = true;
        _begin_cv.notify_all();
    }

    void FeedEnd() {
        std::unique_lock<std::mutex> lk(_mtx);

        _end = true;
        _end_cv.notify_all();
    }

   private:
    void CheckRoutine() {
        pthread_setname_np(pthread_self(), "time_monitor");
        while (_running) {
            std::unique_lock<std::mutex> lk(_mtx);
            _begin_cv.wait(lk, [this]() { return _begin || !_running; });
            if (!_running) {
                return;
            }
            auto begin_time = std::chrono::steady_clock::now();
            _begin = false;

            while (!_end && _running) {
                auto res = _end_cv.wait_for(lk, std::chrono::milliseconds(_period_ms));
                if (res == std::cv_status::timeout) {
                    if (_cb) {
                        auto curr_time = std::chrono::steady_clock::now();
                        uint64_t duration_ms =
                            std::chrono::duration<double, std::milli>(curr_time - begin_time).count();
                        _cb(duration_ms);
                    }
                }
            }
            _end = false;
        }
    }

    uint64_t _period_ms;
    bool _running = true;
    std::unique_ptr<std::thread> _routine;
    std::mutex _mtx;
    std::condition_variable _begin_cv;
    bool _begin = false;
    std::condition_variable _end_cv;
    bool _end = false;
    CallBack _cb;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon