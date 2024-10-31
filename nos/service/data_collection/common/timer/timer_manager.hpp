//
// Created by cheng on 23-7-3.
//

#ifndef TOOLS_TIMER_MANAGER_HPP
#define TOOLS_TIMER_MANAGER_HPP

#include <utility>

#include "common/dc_macros.h"
#include "include/magic_enum.hpp"
#include "thread_pool/include/thread_pool_flex.h"
#include "timer.hpp"
#include "utils/include/time_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

class TimerManager {
   public:
    TimerManager() : stopFlag_(false) {}

    ~TimerManager() {
        stopAll();
    }

    Timer* addTimer(unsigned int timeoutMs, std::function<void(void)> function) {
        Timer* timer = genTimer(TaskPriority::LOW, timeoutMs, 1, 2, std::move(function));
        addTimer(timer);
        return timer;
    }

    /**
     * 添加定时器接口
     * @param priority 当前timer的优先级, 同一时间多个的timer, 高优先级的优先在线程池执行
     * @param timeoutMs 超时时间, 单位 ms, 在timeout毫秒后执行该timer.
     * @param executeTimes 执行次数: 0: 不执行, 1~n: 执行n次, -1: 永远执行.
     * @param intervalMs 多次执行时的执行时间间隔
     * @param function 要执行的函数.
     * @return 创建的Timer对象. 可以timer.setRunCallbackFlag(false) 达到O(1)的删除效果.
     */
    Timer* addTimer(TaskPriority priority, unsigned int timeoutMs, int executeTimes, uint64_t intervalMs, std::function<void(void)> function) {
        Timer* timer = genTimer(priority, timeoutMs, executeTimes, intervalMs, std::move(function));
        addTimer(timer);
        return timer;
    }

    void addTimer(Timer* timer) {
        if (timer != nullptr) {
            std::lock_guard<std::mutex> queueGuard(mtx_);
            queue_.push(timer);
        } else {
            std::cout << "add timer failed!" << std::endl;
        }
    }

    static Timer* genTimer(TaskPriority priority, unsigned int timeoutMs, int executeTimes, uint64_t intervalMs, std::function<void(void)> function) {
        unsigned long long now = getCurrentMillisecs();
        Timer* result;
        DC_NEW(result, Timer(now + timeoutMs, executeTimes, intervalMs, function));
        result->setPara(priorityKey_, std::string(magic_enum::enum_name(priority)));
        return result;
    }

    //    static Timer* genMomentTimer(unsigned long long momentMs, std::function<void(void)> function) {
    //        if (momentMs < getCurrentMillisecs())
    //            return nullptr;
    //        Timer* result;
    //        DC_NEW(result, Timer(momentMs, function))
    //        return result;
    //    }

    /**
     * 删除timer
     * @param timer 删除队列内的timer, timer对象由外部释放.
     */
    void deleteTimer(Timer* timer) {
        std::priority_queue<Timer*, std::vector<Timer*>, cmp> newqueue;
        std::lock_guard<std::mutex> queueGuard(mtx_);
        while (!queue_.empty()) {
            Timer* top = queue_.top();
            queue_.pop();
            if (top != timer)
                newqueue.push(top);
            else {
                DC_DELETE(timer);
            }
        }
        queue_ = newqueue;
    }

    /**
     * 禁用timer
     * @param timer ,禁用timer, timer虽然会被调用,但不会真正执行, 对象由timer manager释放.
     */
    void disableTimer(Timer* timer) {
        deleteTimer(timer);
    }

    long long getRecentTimeout() {
        if (stopFlag_)
            return invalidTimout_;
        long long timeoutMs = invalidTimout_;
        {
            std::lock_guard<std::mutex> queueGuard(mtx_);
            if (queue_.empty())
                return timeoutMs;

            auto nowMs = (long long)getCurrentMillisecs();
            timeoutMs = queue_.top()->getExpire() - nowMs;
        }
        if (timeoutMs < 0)
            timeoutMs = 0;

        return timeoutMs;
    }

    void execTimeoutTimer() {
        unsigned long long now = getCurrentMillisecs();
        std::unique_lock<std::mutex> queueGuard(mtx_);
        while (!queue_.empty() && !stopFlag_) {
            Timer* timer = queue_.top();
            if (timer->getExpire() <= now) {
                queue_.pop();
                queueGuard.unlock();
                auto result = timer->active();
                queueGuard.lock();
                if (!result) {
                    DC_DELETE(timer);
                } else {
                    queue_.push(timer);
                }
                continue;
            }
            return;
        }
    }

//    void start2(ThreadPoolFlex& threadPoolFlex_) {
//        if (taskDemonIsRunning_.load(std::memory_order::memory_order_acquire)) {
//            return;
//        }
//        taskDemonIsRunning_.store(true, std::memory_order::memory_order_release);
//        taskDemon_ = std::thread([&threadPoolFlex_, this] {
//            const long long heatbeatIntervalMs = 100;
//            long long recentTimeout;
//            while (!stopFlag_.load(std::memory_order::memory_order_acquire)) {
//                recentTimeout = getRecentTimeout();
//                if (recentTimeout == 0) {
//                    threadPoolFlex_.add([this] { execTimeoutTimer(); });
//                    continue;
//                }
//                if (recentTimeout > 0) {
//                    long long minSleepTimeMs = std::min(recentTimeout, heatbeatIntervalMs);
//                    TimeUtils::sleepWithWakeup(minSleepTimeMs, stopFlag_);
//                } else {
//                    TimeUtils::sleepWithWakeup(heatbeatIntervalMs, stopFlag_);
//                }
//            }
//        });
//    }

    void start(ThreadPoolFlex& threadPoolFlex_, std::string name="timer") {
        if (taskDemonIsRunning_.load(std::memory_order::memory_order_acquire)) {
            return;
        }
        taskDemonIsRunning_.store(true, std::memory_order::memory_order_release);
        taskDemon_ = std::thread([&threadPoolFlex_, this, name] {
            const int64_t heatbeatIntervalMs = 1000;
            const int64_t threadRunBuffMs = 1;
            pthread_setname_np(pthread_self(), name.substr(0,15).c_str());
            while (!stopFlag_.load(std::memory_order::memory_order_acquire)) {
                Timer* topTimer;
                int64_t timeoutMs;
                std::unique_lock<std::mutex> queueGuard(mtx_);
                if (queue_.empty()) {
                    queueGuard.unlock();
                    TimeUtils::sleepWithWakeup(heatbeatIntervalMs, stopFlag_);
                    continue;
                } else {
                    topTimer = queue_.top();
                    timeoutMs = (int64_t)topTimer->getExpire() - (int64_t)getCurrentMillisecs();
                    if (timeoutMs <= threadRunBuffMs) {
                        queue_.pop();
                    }
                }
                queueGuard.unlock();
                if (timeoutMs <= threadRunBuffMs) {
                    TaskPriority priority = magic_enum::enum_cast<TaskPriority>(topTimer->getPara(priorityKey_)).value();
                    threadPoolFlex_.add(priority, [topTimer, this] {
                        Timer* curTimer = topTimer;
                        auto result = curTimer->active();
                        if (!result) {
                            DC_DELETE(curTimer);
                        } else {
                            std::lock_guard<std::mutex> queueGuard(mtx_);
                            queue_.push(curTimer);
                        }
                    });
                } else {
                    auto minSleepTimeMs = std::min(timeoutMs, heatbeatIntervalMs);
                    TimeUtils::sleepWithWakeup(minSleepTimeMs, stopFlag_);
                }
            }
        });
    }

    static unsigned long long getCurrentMillisecs() {
        struct timespec ts {};
        clock_gettime(CLOCK_REALTIME, &ts);
        return ts.tv_sec * 1000 + ts.tv_nsec / (1000 * 1000);
    }

    void stopAll() {
        if (stopFlag_) {
            return;
        }
        stopFlag_.store(true, std::memory_order::memory_order_release);
        if (taskDemonIsRunning_.load(std::memory_order::memory_order_acquire)) {
            if (taskDemon_.joinable()) {
                taskDemon_.join();
            }
            taskDemonIsRunning_.store(false, std::memory_order::memory_order_release);
        }

        std::lock_guard<std::mutex> queueGuard(mtx_);
        while (!queue_.empty()) {
            Timer* top = queue_.top();
            queue_.pop();
            DC_DELETE(top);
        }
    }

   public:
    const static long long invalidTimout_ = -1;

   private:
    struct cmp {
        bool operator()(Timer*& lhs, Timer*& rhs) const { return lhs->getExpire() > rhs->getExpire(); }
    };

    bool queueEmpty() {
        std::lock_guard<std::mutex> queueGuard(mtx_);
        return queue_.empty();
    }

    std::priority_queue<Timer*, std::vector<Timer*>, cmp> queue_;
    static constexpr const char* priorityKey_{"priority"};
    std::mutex mtx_;
    std::atomic_bool stopFlag_{false};
    std::atomic_bool taskDemonIsRunning_{false};
    std::thread taskDemon_;
};
}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // TOOLS_TIMER_MANAGER_HPP
