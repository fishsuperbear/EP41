// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file thread_safe_queue.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_THREAD_SAFE_QUEUE_H__
#define __LOG_COLLECTOR_INCLUDE_THREAD_SAFE_QUEUE_H__

#include <chrono>             // std::chrono::seconds
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable, std::cv_status

namespace hozon {
namespace netaos {
namespace logcollector {

template <typename T>
class LogQueue {
public:
    LogQueue() {
        array_ = nullptr;
        cap_ = 0;
        size_ = 0;
        front_ = 0;
        tail_ = 0;
    }
    ~LogQueue() {
        destroy();
    }

    int init(int cap) {
        if (cap < 2) {
            return -1;
        }
        array_ = new T[cap];
        if (array_ == nullptr) {
            return -1;
        }
        cap_ = cap;
        clear();

        return 0;
    }

    int push_back(const T &val) {
        if (full()) {
            return -1;
        }
        array_[tail_] = val;
        ++size_;
        if (++tail_ == cap_) {
            tail_ = 0;
        }
        return 0;
    }

    int push_front(const T &val) {
        if (full()) {
            return -1;
        }
        if (val == nullptr) abort();
        array_[front_] = val;
        ++size_;
        if (--front_ < 0) {
            front_ = cap_ - 1;
        }
        return 0;
    }

    int pop_front(T &val) {
        if (empty()) {
            return -1;
        }
        val = array_[front_];
        --size_;
        if (++front_ == cap_) {
            front_ = 0;
        }
        return 0;
    }
    
    int pop_back(T &val) {
        if (empty()) {
            return -1;
        }
        if (--tail_ < 0) {
            tail_ = cap_ - 1;
        }
        val = array_[tail_];
        --size_;
        return 0;
    }

    int pop_backs(T *val, int nums) {
        int count = 0;
        //printf("fuck here....\n");
        while (count < nums) {
            if (pop_back(val[count]) != 0) {
                return count;
            }
            //printf("size:%d front:%d tail:%d val: %lx\n", size_, front_, tail_, val[count]);
            //if (val[count] == nullptr) abort();
            ++count;
        }
        return count;
    }

    int size() const {
        return size_;
    }

    int capacity() const {
        return cap_;
    }

    bool empty() const {
        return size_ == 0;
    }

    bool full() const {
        return size_ == cap_;
    }

    void clear() {
        size_ = 0;
        front_ = 0;
        tail_ = 1;
    }

    void destroy() {
        if (array_) {
            delete []array_;
            array_ = nullptr;
        }
        cap_ = 0;
    }

private:
    LogQueue(const LogQueue &) = delete;
    LogQueue& operator=(const LogQueue &) = delete;

private:
    T *array_;
    int cap_;
    int size_;
    int front_;
    int tail_;
};

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() {}
    ~ThreadSafeQueue() {
        destroy();
    }

    int init(int cap) {
        if (inited()) {
            return -1;
        }

        if (queue_.init(cap) == -1) {
            return -2;
        }

        inited_ = true;
        working_ = true;
        return 0;
    }

    void destroy() {
        if (!inited()) {
            return;
        }
        inited_ = false;
        queue_.destroy();
    }

    bool inited() const {
        return inited_;
    }

    int push_back(const T &val) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!working_) {
            return -1;
        }
        int ret = queue_.push_back(val);
        if (ret == 0) {
            cond_.notify_one();
        }
        return 0;
    }

    int push_front(const T &val) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!working_) {
            return -1;
        }
        int ret = queue_.push_front(val);
        if (ret == 0) {
            cond_.notify_one();
        }
        return 0;
    }

    int pop_front(T &val, int secs) {
        std::unique_lock<std::mutex> lk(mutex_);
        if (!working_) {
            return -1;
        }
        if (queue_.empty() && secs > 0) {
            cond_.wait_for(lk, std::chrono::seconds(secs), [this]{ return !queue_.empty(); });
        }
        return queue_.pop_front(val);
    }

    int pop_back(T &val, int ms) {
        std::unique_lock<std::mutex> lk(mutex_);
        if (!working_) {
            return -1;
        }
        if (queue_.empty() && ms > 0) {
            cond_.wait_for(lk, std::chrono::milliseconds(ms), [this]{ return !queue_.empty(); });
        }
        return queue_.pop_back(val);
    }

    int pop_backs(T *val, int nums, int ms) {
        std::unique_lock<std::mutex> lk(mutex_);
        if (!working_) {
            return -1;
        }
        if (queue_.empty() && ms > 0) {
            cond_.wait_for(lk, std::chrono::milliseconds(ms), [this]{ return !queue_.empty(); });
        }
        //printf ("pop_backs -----> %d\n", queue_.size());
        return queue_.pop_backs(val, nums);
    }

    int size() {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!working_) {
            return -1;
        }
        return queue_.size();
    }

    void stop_work() {
        std::lock_guard<std::mutex> lk(mutex_);
        working_ = false;
        cond_.notify_one();
    }

private:
    bool inited_ = false;
    bool working_ = false;

public:
    LogQueue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};

} // namespace logcollector 
} // namespace netaos
} // namespace hozon 

#endif // __LOG_COLLECTOR_INCLUDE_THREAD_SAFE_QUEUE_H__
