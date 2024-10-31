/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: thread_pool_flex.h
 * @Date: 2023/08/16
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef MIDDLEWARE_TOOLS_COMMON_THREAD_POOL_THREAD_POOL_FLEX_H_
#define MIDDLEWARE_TOOLS_COMMON_THREAD_POOL_THREAD_POOL_FLEX_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "thread_guard.h"
#include "basic/task.h"

namespace hozon {
namespace netaos {
namespace dc {


typedef std::function<void()> TaskType;

struct ThreadTask {
    TaskPriority taskPriority;
    TaskType task;

    ThreadTask(TaskPriority priority, TaskType taskType) {
        this->taskPriority = priority;
        this->task = std::move(taskType);
    }

    ThreadTask() = default;

    bool operator<(const ThreadTask& a) const { return taskPriority < a.taskPriority; }
};

/**
 * 任务可排序, 线程数量可变化.
 */
class ThreadPoolFlex {
   public:
    explicit ThreadPoolFlex(int runningThreadNum = 0);

    explicit ThreadPoolFlex(int runningThreadNum, int maxThreadNum);

    ~ThreadPoolFlex();

    /**
     * 停止线程池
     */
    void stop();

    /**
     * 在运行的task数量
     */
    int getRunningTasks();

    /**
     * 添加带优先级的任务
     * @tparam Function 要执行的函数
     * @tparam Args 函数的入参
     * @param priority 优先级 TaskPriority
     * @return  future
     */
    template <class Function, class... Args>
    std::future<typename std::result_of<Function(Args...)>::type> add(TaskPriority priority, Function&&, Args&&...);

    /**
     * 添加默认优先级(LOW)的任务
     * @tparam Function 要执行的函数
     * @tparam Args 函数的入参
     * @param priority 优先级 TaskPriority
     * @return  future
     */
    template <class Function, class... Args>
    std::future<typename std::result_of<Function(Args...)>::type> add(Function&&, Args&&...);

    int getQueueSize() {
        std::lock_guard<std::mutex> lg(mtx_);
        return tasks_.size();
    }

   private:
    ThreadPoolFlex(ThreadPoolFlex&&) = delete;

    ThreadPoolFlex& operator=(ThreadPoolFlex&&) = delete;

    ThreadPoolFlex(const ThreadPoolFlex&) = delete;

    ThreadPoolFlex& operator=(const ThreadPoolFlex&) = delete;

    void start();

    void startExtra();

   private:
    std::atomic<bool> stop_;
    std::mutex mtx_;
    std::condition_variable cond_;

    std::priority_queue<ThreadTask> tasks_;
    std::vector<std::thread> threads_;
    ThreadsGuard tg_;

    int runningThreadNum_;
    const int maxThreadNum_;
    std::atomic_int runningTasks_;
};

template <class Function, class... Args>
std::future<typename std::result_of<Function(Args...)>::type> ThreadPoolFlex::add(TaskPriority priority, Function&& fcn, Args&&... args) {
    typedef typename std::result_of<Function(Args...)>::type return_type;
    typedef std::packaged_task<return_type()> task;

    auto t = std::make_shared<task>(std::bind(std::forward<Function>(fcn), std::forward<Args>(args)...));
    auto ret = t->get_future();
    bool notifyFlag = false;
    {
        std::lock_guard<std::mutex> lg(mtx_);
        if (stop_.load(std::memory_order_acquire))
            throw std::runtime_error("thread pool has been stopped");
        if (priority != TaskPriority::EXTRA_HIGH || runningTasks_ < runningThreadNum_ || runningTasks_ >= maxThreadNum_) {
            tasks_.push({priority, [t] {
                             (*t)();
                         }});
            notifyFlag = true;
        } else {
            threads_.emplace_back([t, this] {
                runningTasks_++;
                (*t)();
                runningTasks_--;
                while (!stop_.load(std::memory_order_acquire)) {
                    ThreadTask currentTask;
                    {
                        std::unique_lock<std::mutex> ulk(this->mtx_);
                        this->cond_.wait(ulk, [this] { return stop_.load(std::memory_order_acquire) || !this->tasks_.empty(); });
                        currentTask = this->tasks_.top();
                        if (currentTask.taskPriority != TaskPriority::EXTRA_HIGH) {
                            break;
                        }
                        this->tasks_.pop();
                    }
                    runningTasks_++;
                    currentTask.task();
                    runningTasks_--;
                }
            });
        }
    }
    if (notifyFlag) {
        cond_.notify_one();
    }
    return ret;
}

template <class Function, class... Args>
std::future<typename std::result_of<Function(Args...)>::type> ThreadPoolFlex::add(Function&& fcn, Args&&... args) {
    return add(TaskPriority::LOW, fcn, args...);
}


}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_COMMON_THREAD_POOL_THREAD_POOL_FLEX_H_
