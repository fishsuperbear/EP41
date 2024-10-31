/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip thread
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_THREAD_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_THREAD_H_

#include <pthread.h>
#include <stdint.h>
#include <string>
#include <mutex>
#include <functional>


namespace hozon {
namespace netaos {
namespace diag {


using thread_callback = std::function<void(void*)>;


typedef struct doip_thread {
    pthread_t thread;
    pid_t tid;
    std::string name;
    void *arg;
    int32_t quit_flag;
    pthread_attr_t attr;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_condattr_t condattr;
    thread_callback main_fun;
} doip_thread_t;

typedef struct doip_sync {
    pthread_mutex_t mutex;
    pthread_mutexattr_t attr;
} doip_sync_t;

typedef struct doip_cond {
    pthread_cond_t cond;
    pthread_condattr_t condattr;
    pthread_mutex_t lock;
} doip_cond_t;


class DoipThread {
 public:
    static DoipThread *Instance();
    doip_thread_t* DoipThreadCreate(thread_callback main_fun, void *arg, std::string name);
    void DoipThreadRelease(doip_thread_t *thread);
    std::string DoipThreadGetName(const doip_thread_t *thread);
    int32_t DoipThreadGetId(doip_thread_t *thread);
    int32_t DoipThreadWait(doip_thread_t *thread);
    int32_t DoipThreadWaitTimeout(doip_thread_t *thread, int32_t timeout_ms);
    int32_t DoipThreadNotify(doip_thread_t *thread);
    int32_t DoipThreadJoin(doip_thread_t *thread);
    int32_t DoipThreadTerminate(doip_thread_t *thread);
    int32_t DoipThreadStop(doip_thread_t *thread);
    int32_t DoipThreadCheckquit(doip_thread_t *thread);

 private:
    DoipThread();
    ~DoipThread();
    DoipThread(const DoipThread &);
    DoipThread & operator = (const DoipThread &);
    static void* DoipThreadMain(void *arg);

    static DoipThread *instancePtr_;
    static std::mutex instance_mtx_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_THREAD_H_
