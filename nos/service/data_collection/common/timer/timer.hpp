//
// Created by cheng on 23-6-30.
//

#pragma once
#ifndef TOOLS_TIMER_H
#define TOOLS_TIMER_H

#include <sys/epoll.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <vector>

struct TimingConfiguration{
    unsigned int timeOutMs = 0;
    int executeTimes = 1;
    uint64_t intervalMs = 1000;
    bool autoStop{true};
};
YCS_ADD_STRUCT(TimingConfiguration, timeOutMs, executeTimes, intervalMs, autoStop)

class Timer {
   public:
    Timer(unsigned long long expiredMomentMs, std::function<void(void)>& callback) : Timer(expiredMomentMs, 1, 0, callback) {}

    /**
     * 定时器任务
     * @param expiredMomentMs 失效时间 = timeout+ 当前时间,单位Ms,
     * @param executeTimes 要执行的次数, -1: 永远执行(executedCount_防止溢出),
     * @param intervalMs 执行间隔, 单位Ms,
     * @param callback 回调函数
     */
    Timer(unsigned long long expiredMomentMs, int executeTimes, uint64_t intervalMs, std::function<void(void)>& callback)
        : callback_(callback), expiredMomentMs_(expiredMomentMs), runFlag_(true), executTimes_(executeTimes), executedCount_(0), intervalMs_(intervalMs) {}

    inline bool active() {
        if (executTimes_ == 0) {
            return false;
        }
        executedCount_++;
        executTimes_--;
        if (executTimes_ < -10000) {
            executTimes_ = -1;
        }
        if (executedCount_ > 864000) {
            // 防止溢出
            executedCount_ = 1;
        }
        if (runFlag_) {
            callback_();
        }
        struct timespec ts {};

        clock_gettime(CLOCK_REALTIME, &ts);
        unsigned long long curTime = ts.tv_sec * 1000 + ts.tv_nsec / (1000 * 1000);
        if (timeAbs(expiredMomentMs_ - curTime) > 10000) {
            expiredMomentMs_ = curTime; // 时间跳变矫正.
        }
        expiredMomentMs_ += intervalMs_;
        if (executTimes_ == 0) {
            return false;
        }
        return true;
    }

    inline void setRunCallbackFlag(const bool& flag) { runFlag_ = flag; }

    inline unsigned long long getExpire() const { return expiredMomentMs_; }

    inline void setExpire(const unsigned long long& expire) { expiredMomentMs_ = expire; }

    inline void setPara(const std::string& key, const std::string& value) { paraRec_[key] = value; }

    inline std::string getPara(const std::string& key) { return paraRec_[key]; }

   private:
    std::function<void(void)> callback_;
    std::map<std::string, std::string> paraRec_;
    unsigned long long expiredMomentMs_;
    bool runFlag_ = true;

    unsigned long long timeAbs(long long input) {
        if (input>=0) return input;
        return -input;
    }
    int executTimes_;
    uint32_t executedCount_;

    uint64_t intervalMs_;
};

#endif  //TOOLS_TIMER_H
