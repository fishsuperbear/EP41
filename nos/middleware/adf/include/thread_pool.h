#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace hozon {
namespace netaos {
namespace adf {
class ThreadPool {
   public:
    ThreadPool(uint32_t num) {
        for (uint32_t i = 0; i < num; ++i) {
            _workers.emplace_back([this]() {
                while (!_need_stop) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lk(_mtx);
                        while (_tasks.empty()) {
                            _cv.wait(lk);
                            if (_need_stop) {
                                return;
                            }
                        }

                        task = _tasks.front();
                        _tasks.pop();
                    }

                    task();
                }
            });
            pthread_setname_np(_workers.back().native_handle(),
                               (std::string("thread_pool_") + std::to_string(i)).c_str());
        }
    }

    ~ThreadPool() {}

    void Stop() {
        _need_stop = true;
        _cv.notify_all();
        for (auto& th : _workers) {
            th.join();
        }
    }

    template <class Fn, class... Args>
    auto Commit(Fn&& func, Args&&... args) -> std::future<decltype(func(args...))> {
        std::lock_guard<std::mutex> lk(_mtx);

        std::function<decltype(func(args...))()> fn = std::bind(std::forward<Fn>(func), std::forward<Args>(args)...);
        auto task_ptr = std::make_shared<std::packaged_task<decltype(func(args...))()>>(fn);
        _tasks.emplace([task_ptr]() { (*task_ptr)(); });

        _cv.notify_one();

        return task_ptr->get_future();
    }

   private:
    bool _need_stop = false;
    std::mutex _mtx;
    std::condition_variable _cv;
    std::queue<std::function<void()>> _tasks;
    std::vector<std::thread> _workers;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon