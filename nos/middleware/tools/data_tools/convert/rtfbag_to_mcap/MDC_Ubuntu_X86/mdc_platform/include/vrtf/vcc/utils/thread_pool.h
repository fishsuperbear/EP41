/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Implement thread pool
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_THREAD_POOL_H
#define VRTF_VCC_THREAD_POOL_H
#include <vector>
#include <queue>
#include <set>
#include <list>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include "ara/hwcommon/log/log.h"
#include "vrtf/vcc/utils/lock_free_queue.h"
namespace vrtf {
namespace vcc {
namespace utils {
struct TaskSoureceId {
    uint16_t typeId = 0U;
    uint64_t uuidHight = 0U;
    uint64_t uuidLow = 0U;
    bool operator < (const TaskSoureceId &other) const
    {
        if (typeId != other.typeId) {
            return typeId < other.typeId ;
        } else if (uuidLow != other.uuidLow) {
            return uuidLow < other.uuidLow;
        } else {
            return uuidHight < other.uuidHight;
        }
    }
};

class ThreadPool {
public:
    explicit ThreadPool(size_t size, uint32_t queueSize = 1024U, const std::string threadName = "");
    void TaskEntry();
    bool Enqueue(const std::function<void()> &func, TaskSoureceId const &taskKey = TaskSoureceId());
    ~ThreadPool();
    void Stop() noexcept;
private:
    class ThreadPoolTask;
    ThreadPoolTask PopMultiThreadTask();
    void SingleThreadTaskEntry();
    void MultiThreadTaskEntry();
    bool AllWaitTaskIsRunning() const;
    void PrintLog(size_t taskSize);
    bool EnqueueSingleThreadTask(const std::function<void()> &func);
    bool EnqueueMultiThreadTask(const std::function<void()> &func, TaskSoureceId const &taskKey);
    std::vector<std::thread> workers_;
    std::shared_ptr<LockFreeQueue<std::function<void()>>> singleThreadTask_;
    std::list<ThreadPoolTask> multiThreadTask_;
    std::set<TaskSoureceId> runTask_;
    std::map<TaskSoureceId, uint16_t> waitRunTaskIdList;
    std::mutex queueMutex_;
    std::condition_variable cond_;
    bool isStop_;
    size_t threadSize_;
    uint32_t queueSize_;
    std::string threadName_;
    timespec lastFullTime_ = {0, 0};
    size_t lostTaskNum_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    class ThreadPoolTask {
        public:
            ThreadPoolTask() = default;
            ThreadPoolTask(const TaskSoureceId& key, const std::function<void()>& task)
                : key_(key), task_(task){}
            ThreadPoolTask(const ThreadPoolTask &) = default;
            ThreadPoolTask& operator=(const ThreadPoolTask&) = default;
            ThreadPoolTask(ThreadPoolTask&&) = default;
            ThreadPoolTask& operator=(ThreadPoolTask&&) = default;
            ~ThreadPoolTask() = default;
            explicit operator bool() const { return static_cast<bool>(task_); }
            void operator()() const
            {
                task_();
            }
            TaskSoureceId GetTaskSoureceId() { return key_; }
        private:
            TaskSoureceId key_;
            std::function<void()> task_ {nullptr};
    };
};
}
}
}
#endif
