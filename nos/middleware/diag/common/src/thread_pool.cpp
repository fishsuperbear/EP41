
#include "diag/common/include/thread_pool.h"
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>

namespace hozon {
namespace netaos {
namespace diag {

#define TASK_SIZE_MAX 100

std::vector<BaseTask*> ThreadPool::veBaseTaskList_;
bool ThreadPool::shutdown_ = false;
uint8_t ThreadPool::cidStatArray_[20] = {0};
pthread_mutex_t ThreadPool::pthreadMutex_ = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t ThreadPool::pthreadCond_ = PTHREAD_COND_INITIALIZER;


void BaseTask::setData(void* data)
{
    ptrData_ = data;
}

uint32_t BaseTask::getDependentCID()
{
    return dependentCID_;
}

void BaseTask::setDependentCID(uint32_t cid)
{
    dependentCID_ = cid;
}

uint32_t BaseTask::getMyCID()
{
    return myCID_;
}

void BaseTask::setMyCID(uint32_t cid)
{
    myCID_ = cid;
}

BaseTask::~BaseTask()
{
}


ThreadPool::ThreadPool(int threadNum)
{
    veBaseTaskList_.clear();
    memset(cidStatArray_, 0, 20);
    shutdown_ = false;

    this->iThreadNum_ = threadNum;
    Create();
}

ThreadPool::~ThreadPool()
{
}

int ThreadPool::Create()
{
    pthread_id_ = new pthread_t[iThreadNum_];
    for(int i = 0; i < iThreadNum_; i++) {
        pthread_create(&pthread_id_[i], NULL, ThreadFunc, &i);
    }

    return 0;
}

void* ThreadPool::ThreadFunc(void *tid)
{
    int* id = (int*)(tid);
    std::string thread_name = "thread_pool_tid" + std::to_string(*id);
    pthread_setname_np(pthread_self(), thread_name.c_str());
    while (!shutdown_) {
        pthread_mutex_lock(&pthreadMutex_);
        while (veBaseTaskList_.size() == 0 && !shutdown_) {
            pthread_cond_wait(&pthreadCond_, &pthreadMutex_);
        }

        if (shutdown_) {
            pthread_mutex_unlock(&pthreadMutex_);
            break;
        }

        std::vector<BaseTask*>::iterator iter = veBaseTaskList_.begin();
        BaseTask* task = *iter;
        veBaseTaskList_.erase(iter);

        uint32_t cid1 = task->getDependentCID();
        uint32_t cid2 = task->getMyCID();

        pthread_mutex_unlock(&pthreadMutex_);

        cidStatArray_[cid2%20] = 0;

        if (cid1 > 0) {
            if (cidStatArray_[cid1%20] == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        task->Run();

        cidStatArray_[cid2%20] = 1;

        delete task;
        task = nullptr;
    }

    return (void*)0;
}

int ThreadPool::AddTask(BaseTask *task)
{
    static uint32_t cid = 0;
    if (veBaseTaskList_.size() >= TASK_SIZE_MAX) {
        return -1;
    }
    pthread_mutex_lock(&pthreadMutex_);
    task->setMyCID(++cid);
    veBaseTaskList_.push_back(task);

    pthread_cond_signal(&pthreadCond_);
    pthread_mutex_unlock(&pthreadMutex_);

    if (cid > 20) {
        if ((cid%20)>=0 && (cid%20)<10) {
            memset(cidStatArray_, 0, 10);
        }
        else {
            memset(cidStatArray_+10, 0, 10);
        }
    }

    return cid;
}

int ThreadPool::StopAll()
{
    if (shutdown_) {
        return -1;
    }

    shutdown_ = true;
    pthread_mutex_lock(&pthreadMutex_);
    pthread_cond_broadcast(&pthreadCond_);
    pthread_mutex_unlock(&pthreadMutex_);
    pthread_mutex_destroy(&pthreadMutex_);
    pthread_cond_destroy(&pthreadCond_);

    for (int i = 0; i < iThreadNum_; i++) {
        pthread_join(pthread_id_[i], NULL);
    }

    delete[] pthread_id_;
    pthread_id_ = NULL;

    veBaseTaskList_.clear();

    return 0;
}

int ThreadPool::GetTaskSize()
{
    return veBaseTaskList_.size();
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */