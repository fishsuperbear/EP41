/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip thread pool
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <string>
#include <pthread.h>


namespace hozon {
namespace netaos {
namespace diag {

class BaseTask {
public:
    BaseTask() = default;
    BaseTask(std::string &taskName): strTaskName_(taskName), ptrData_(NULL) {}
    BaseTask(const BaseTask&) = delete;
    BaseTask &operator=(const BaseTask&) = delete;
    void setData(void* data);
    uint32_t getDependentCID();
    void setDependentCID(uint32_t cid);
    uint32_t getMyCID();
    void setMyCID(uint32_t cid);
    virtual int Run() = 0;
    virtual ~BaseTask();

protected:
    std::string strTaskName_;
    void* ptrData_;
    uint32_t myCID_{0};
    uint32_t dependentCID_{0};
};

class ThreadPool {
public:
    ThreadPool(int threadNum);
     ~ThreadPool();
    int AddTask(BaseTask *task);
    int StopAll();
    int GetTaskSize();

protected:
    static void* ThreadFunc(void *tid);
    int Create();


private:
    static std::vector<BaseTask*> veBaseTaskList_;
    static bool shutdown_;
    static pthread_mutex_t pthreadMutex_;
    static uint8_t cidStatArray_[20];
    static pthread_cond_t pthreadCond_;

    int iThreadNum_;
    pthread_t *pthread_id_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // THREAD_POOL_H