//
// Created by cheng on 23-7-3.
//

#ifndef TOOLS_TIMER_TASK_HPP
#define TOOLS_TIMER_TASK_HPP

#include <functional>
#include <memory>
#include <utility>
#include "timer_manager.hpp"

struct TimerTaskBAK {
    std::function<void(void)> function_;
    unsigned int intervalMs_;
    int executTimes_;
    TimerManager* tm_;
    unsigned long long expiredMomentMs_{};

   public:
    /**
     * 设置定时器任务
     * @param function 要执行的函数
     * @param timerManager 定时器管理指针
     * @param timeOutMs 等待timeout 毫秒后执行
     * @param intervalMs 执行间隔,单位Ms.
     * @param executTimes 执行次数( <0: 永远执行, =0: 不执行, >0: 执行n次)
     */
    TimerTaskBAK(std::function<void(void)> function, TimerManager* timerManager, unsigned int timeOutMs, unsigned int intervalMs = 1000, int executTimes = 1) noexcept
        : function_(std::move(function)), intervalMs_(intervalMs), executTimes_(executTimes), tm_(timerManager) {
        if (executTimes != 0 && intervalMs > 0) {
            Timer* timer = tm_->addTimer(timeOutMs, TimerTaskBAK(tm_, function_, intervalMs_, executTimes_, timeOutMs + TimerManager::getCurrentMillisecs()));
            expiredMomentMs_ = timer->getExpire();
        } else {
            //          DEBUG_LOG( "TimerTask input param error" );
        }
    }

    ~TimerTaskBAK() {
        //      DEBUG_LOG("after timer task destruct|\n");
    }

    void operator()() noexcept {
        if (executTimes_ == 0) {
            return;
        }
        if (--executTimes_ != 0) {
            if (executTimes_ < -10000) {
                executTimes_ = -1;
            }
            expiredMomentMs_ += intervalMs_;
            Timer* timer = TimerManager::genMomentTimer(expiredMomentMs_, TimerTaskBAK(tm_, function_, intervalMs_, executTimes_, expiredMomentMs_));
            tm_->addTimer(timer);
        }
        function_();
    }

   private:
    TimerTaskBAK(TimerManager* tm, std::function<void(void)>& function, unsigned int intervalMs, int executTimes, unsigned long long expiredMomentMs) noexcept
        : function_(function), intervalMs_(intervalMs), executTimes_(executTimes), tm_(tm), expiredMomentMs_(expiredMomentMs) {}
};

#endif  //TOOLS_TIMER_TASK_HPP
