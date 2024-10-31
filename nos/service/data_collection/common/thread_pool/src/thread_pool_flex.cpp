/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: thread_pool_flex.cpp
 * @Date: 2023/08/16
 * @Author: cheng
 * @Desc: --
 */

#include "thread_pool/include/thread_pool_flex.h"
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

#include "thread_pool/include/thread_guard.h"

namespace hozon {
namespace netaos {
namespace dc {
    ThreadPoolFlex::~ThreadPoolFlex() {
        stop();
        cond_.notify_all();
    }

    /**
     * 停止线程池
     */
    void ThreadPoolFlex::stop() {
        stop_.store(true, std::memory_order_release);
    }

    /**
     * 在运行的task数量
     */
    int ThreadPoolFlex::getRunningTasks() {
        return runningTasks_;
    }


ThreadPoolFlex::ThreadPoolFlex(int runningThreadNum)
    : ThreadPoolFlex(runningThreadNum, runningThreadNum) {
}

ThreadPoolFlex::ThreadPoolFlex(int runningThreadNum, int maxThreadNum)
    : stop_(false), tg_(threads_), maxThreadNum_(maxThreadNum) {
    int nthreads = runningThreadNum;
    if (nthreads <= 0) {
        nthreads = std::thread::hardware_concurrency();
        nthreads = (nthreads == 0 ? 2 : nthreads);
    }
    runningThreadNum_ = nthreads;
    runningTasks_ = 0;
    // max running thread number when busy. not implement.
    start();
}

void ThreadPoolFlex::start() {
    for (int i = 0; i < runningThreadNum_; ++i) {
        threads_.emplace_back([this] {
            pthread_setname_np(pthread_self(), "thread_pool");
            while (!stop_.load(std::memory_order_acquire)) {
                ThreadTask currentTask;
                {
                    std::unique_lock<std::mutex> ulk(this->mtx_);
                    this->cond_.wait(ulk, [this] {
                        return stop_.load(std::memory_order_acquire) || !this->tasks_.empty();
                    });
                    if (stop_.load(std::memory_order_acquire))
                        return;
                    currentTask = this->tasks_.top();
                    this->tasks_.pop();
                }
                runningTasks_++;
                pthread_setname_np(pthread_self(), "thread_pool");
                currentTask.task();
                pthread_setname_np(pthread_self(), "thread_pool");
                runningTasks_--;
            }
        });
    }
}

//void ThreadPoolFlex::startExtra() {
//    ThreadTask currentTask;
//    {
//        std::lock_guard<std::mutex> lockGuard(mtx_);
//        if (tasks_.empty()) return;
//    }
//    for (int i = runningThreadNum_; i < maxThreadNum_; ++i) {
//        threads_.emplace_back([this] {
//            while (!stop_.load(std::memory_order_acquire)) {
//                ThreadTask currentTask;
//                {
//                    std::unique_lock<std::mutex> ulk(this->mtx_);
//                    this->cond_.wait(ulk, [this] {
//                        return stop_.load(std::memory_order_acquire) || !this->tasks_.empty();
//                    });
//                    if (stop_.load(std::memory_order_acquire))
//                        return;
//                    currentTask = this->tasks_.top();
//                    this->tasks_.pop();
//                }
//                runningTasks_++;
//                currentTask.task();
//                runningTasks_--;
//            }
//        });
//    }
//}
//
//template<class Function, class... Args>
//std::future<typename std::result_of<Function(Args...)>::type>
//ThreadPoolFlex::add(TaskPriority priority, Function &&fcn, Args &&... args) {
//    typedef typename std::result_of<Function(Args...)>::type return_type;
//    typedef std::packaged_task<return_type()> task;
//
//    auto t = std::make_shared<task>(std::bind(std::forward<Function>(fcn), std::forward<Args>(args)...));
//    auto ret = t->get_future();
//    {
//        std::lock_guard<std::mutex> lg(mtx_);
//        if (stop_.load(std::memory_order_acquire))
//            throw std::runtime_error("thread pool has stopped");
//        tasks_.push({priority, [t] { (*t)(); }});
//    }
//    cond_.notify_one();
//    return ret;
//}
//
//template<class Function, class... Args>
//std::future<typename std::result_of<Function(Args...)>::type>
//ThreadPoolFlex::add(Function &&fcn, Args &&... args) {
//    return add(TaskPriority::LOW, fcn, args...);
//}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
