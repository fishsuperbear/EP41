/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip thread
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <functional>

#include "diag/doip/include/base/doip_thread.h"

namespace hozon {
namespace netaos {
namespace diag {


DoipThread *DoipThread::instancePtr_ = nullptr;
std::mutex DoipThread::instance_mtx_;

DoipThread *DoipThread::Instance() {
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new DoipThread();
        }
    }
    return instancePtr_;
}

DoipThread::DoipThread() {
}

DoipThread::~DoipThread() {
}

doip_thread_t*
DoipThread::DoipThreadCreate(thread_callback main_fun, void *arg, std::string name) {
    if ((main_fun == nullptr) || (name == "")) {
        return nullptr;
    }

    doip_thread_t *thread = new doip_thread_t;
    thread->quit_flag = 0;
    thread->arg = arg;
    thread->main_fun = main_fun;
    thread->thread = -1;
    thread->tid = -1;
    thread->name = name;

    int32_t ret = -1;
    ret = pthread_condattr_init(&thread->condattr);
    if (ret != 0) {
        delete thread;
        return nullptr;
    }

    ret = pthread_condattr_setclock(&thread->condattr, CLOCK_MONOTONIC);
    if (ret != 0) {
        pthread_condattr_destroy(&thread->condattr);
        delete thread;
        return nullptr;
    }

    ret = pthread_cond_init(&thread->cond, &thread->condattr);
    if (ret != 0) {
        pthread_condattr_destroy(&thread->condattr);
        delete thread;
        return nullptr;
    }

    ret = pthread_mutex_init(&thread->lock, NULL);
    if (ret != 0) {
        pthread_condattr_destroy(&thread->condattr);
        pthread_cond_destroy(&thread->cond);
        delete thread;
        return nullptr;
    }

    ret = pthread_attr_init(&thread->attr);
    if (0 != ret) {
        pthread_mutex_destroy(&thread->lock);
        pthread_condattr_destroy(&thread->condattr);
        pthread_cond_destroy(&thread->cond);
        delete thread;
        return nullptr;
    }

    ret = pthread_attr_setscope(&thread->attr, PTHREAD_SCOPE_SYSTEM);
    if (0 != ret) {
        pthread_attr_destroy(&thread->attr);
        pthread_mutex_destroy(&thread->lock);
        pthread_condattr_destroy(&thread->condattr);
        pthread_cond_destroy(&thread->cond);
        delete thread;
        return nullptr;
    }

    ret = pthread_create(&thread->thread, &thread->attr, DoipThread::DoipThreadMain, thread);
    if (ret != 0) {
        pthread_attr_destroy(&thread->attr);
        pthread_mutex_destroy(&thread->lock);
        pthread_condattr_destroy(&thread->condattr);
        pthread_cond_destroy(&thread->cond);
        delete thread;
        return nullptr;
    }

    return thread;
}

void
DoipThread::DoipThreadRelease(doip_thread_t *thread) {
    if (nullptr == thread) {
        return;
    }
    pthread_cond_destroy(&thread->cond);
    pthread_mutex_destroy(&thread->lock);
    pthread_condattr_destroy(&thread->condattr);
    pthread_attr_destroy(&thread->attr);

    delete thread;
    thread = nullptr;
}

std::string
DoipThread::DoipThreadGetName(const doip_thread_t *thread) {
    if (nullptr == thread) {
        return "unknown";
    }
    return thread->name;
}

int32_t
DoipThread::DoipThreadGetId(doip_thread_t *thread) {
    if (nullptr == thread) {
        return -1;
    }
    return thread->tid;
}

int32_t
DoipThread::DoipThreadWait(doip_thread_t *thread) {
    if (nullptr == thread) {
        return -1;
    }
    pthread_mutex_lock(&thread->lock);
    int32_t ret = pthread_cond_wait(&thread->cond, &thread->lock);
    pthread_mutex_unlock(&thread->lock);

    return ret;
}

int32_t
DoipThread::DoipThreadWaitTimeout(doip_thread_t *thread, int32_t timeout_ms) {
    if (nullptr == thread) {
        return -1;
    }
    struct timespec nptime;

    pthread_mutex_lock(&thread->lock);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    nptime.tv_sec = now.tv_sec + (timeout_ms / 1000);
    nptime.tv_nsec = now.tv_nsec + (timeout_ms % 1000) * 1000000;

    /* bigger than one second */
    if (nptime.tv_nsec >= 1000000000) {
        nptime.tv_sec++;
        nptime.tv_nsec -= 1000000000;
    }
    int32_t ret = pthread_cond_timedwait(&thread->cond, &thread->lock, &nptime);
    pthread_mutex_unlock(&thread->lock);

    return ret;
}

int32_t
DoipThread::DoipThreadNotify(doip_thread_t *thread) {
    if (nullptr == thread) {
        return -1;
    }
    return pthread_cond_signal(&thread->cond);
}

int32_t
DoipThread::DoipThreadJoin(doip_thread_t *thread) {
    if (nullptr == thread) {
        return -1;
    }
    return pthread_join(thread->thread, NULL);
}

int32_t
DoipThread::DoipThreadTerminate(doip_thread_t *thread) {
    if (nullptr == thread) {
        return -1;
    }

    thread->quit_flag = 1;

    int32_t ret = pthread_cancel(thread->thread);
    if (ret == 0) {
        ret = DoipThreadJoin(thread);
    }

    DoipThreadRelease(thread);

    return ret;
}

int32_t
DoipThread::DoipThreadStop(doip_thread_t *thread) {
    if (nullptr == thread) {
        return -1;
    }

    thread->quit_flag = 1;

    int32_t ret = DoipThreadNotify(thread);
    if (ret == 0) {
        ret = DoipThreadJoin(thread);
    }

    DoipThreadRelease(thread);

    return ret;
}

int32_t
DoipThread::DoipThreadCheckquit(doip_thread_t *thread) {
    if (nullptr == thread) {
        return 1;
    }
    return thread->quit_flag;
}

void*
DoipThread::DoipThreadMain(void *arg) {
    doip_thread_t *thread = reinterpret_cast<doip_thread_t *>(arg);
    if (nullptr == thread) {
        return nullptr;
    }
    thread->tid = getpid();
    thread->main_fun(thread->arg);

    return nullptr;
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
