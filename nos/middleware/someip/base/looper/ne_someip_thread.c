/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "ne_someip_object.h"
#include "ne_someip_thread.h"
#include "ne_someip_log.h"


/* ne_someip_thread对象在TLS中存储的key */
static pthread_key_t g_tls_key_thread;
/* TLS中已经存储了NELooper？ 0：没有 1：有 */
static int32_t g_tls_exist;
/* TLS读写锁 */
static pthread_mutex_t g_tls_mutex;

// ================== start ne_someip_thread_t object ==============
#define NE_SOMEIP_THREAD_NAME_SIZE 16
struct ne_someip_thread {
    pthread_t id;                        // 线程id
    void* user_data;                    // 主函数参数
    int32_t is_running;                        // 线程运行状态。初始为0，在ne_someip_threadStart成功时置为1，ne_someip_threadStop复位为0.
    ne_someip_thread_user_main_func thread_func;    // 线程主函数
    ne_someip_thread_user_free_func free_func;        // 用户资源释放函数

    char name[NE_SOMEIP_THREAD_NAME_SIZE + 1];
    int32_t signal_flag;
    pthread_attr_t attr;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_condattr_t condattr;

    bool looper_run;

    NEOBJECT_MEMBER
};
void ne_someip_thread_t_free(ne_someip_thread_t* thread);
NEOBJECT_FUNCTION(ne_someip_thread_t);
ne_someip_thread_t* ne_someip_thread_t_new()
{
    ne_someip_thread_t* thread = malloc(sizeof(ne_someip_thread_t));
    if (!thread) {
        ne_someip_log_error("[Thread] malloc error");
        return thread;
    }

    memset(thread, 0, sizeof(ne_someip_thread_t));

    int32_t ret = -1;
    ret = pthread_condattr_init(&thread->condattr);
    if (ret != 0) {
        ne_someip_log_error("[Thread] pthread_condattr_init error");
        free(thread);
        thread = NULL;
        return NULL;
    }

    ret = pthread_condattr_setclock(&thread->condattr, CLOCK_MONOTONIC);
    if (ret != 0) {
        ne_someip_log_error("[Thread] pthread_condattr_setclock error");
        pthread_condattr_destroy(&thread->condattr);
        free(thread);
        thread = NULL;
        return NULL;
    }


    ret = pthread_cond_init(&thread->cond, &thread->condattr);
    if (ret != 0) {
        ne_someip_log_error("[Thread] pthread_cond_init error");
        pthread_condattr_destroy(&thread->condattr);
        free(thread);
        thread = NULL;
        return NULL;
    }

    ret = pthread_mutex_init(&thread->lock, NULL);
    if (ret != 0) {
        ne_someip_log_error("[Thread] pthread_mutex_init error");
        pthread_condattr_destroy(&thread->condattr);
        pthread_cond_destroy(&thread->cond);
        free(thread);
        thread = NULL;
        return NULL;
    }

    ret = pthread_attr_init(&thread->attr);
    if (0 != ret) {
        ne_someip_log_error("[Thread] pthread_attr_init error");
        pthread_mutex_destroy(&thread->lock);
        pthread_condattr_destroy(&thread->condattr);
        pthread_cond_destroy(&thread->cond);
        free(thread);
        thread = NULL;
        return NULL;
    }

    ret = pthread_attr_setscope(&thread->attr, PTHREAD_SCOPE_SYSTEM);
    if (0 != ret) {
        ne_someip_log_error("[Thread] pthread_attr_setscope error");
        pthread_attr_destroy(&thread->attr);
        pthread_mutex_destroy(&thread->lock);
        pthread_condattr_destroy(&thread->condattr);
        pthread_cond_destroy(&thread->cond);
        free(thread);
        thread = NULL;
        return NULL;
    }

    ne_someip_thread_t_ref_count_init(thread);

    ne_someip_log_debug("[Thread] create thread %p", thread);

    return thread;
}

void ne_someip_thread_t_free(ne_someip_thread_t* thread)
{
    if (!thread) {
        ne_someip_log_error("[Thread] param is null");
        return;
    }

    ne_someip_log_debug("[Thread] free thread %p", thread);

    ne_someip_thread_t_ref_count_deinit(thread);

    if (thread->free_func) {
        thread->free_func(thread->user_data);
    }

    if (thread->looper_run) {
        // 当前thread为looper专用，user data为looper，thread退出，引用-1
        ne_someip_looper_unref((ne_someip_looper_t*)thread->user_data);
    }

    pthread_cond_destroy(&thread->cond);
    pthread_mutex_destroy(&thread->lock);
    pthread_condattr_destroy(&thread->condattr);
    pthread_attr_destroy(&thread->attr);

    free(thread);
    thread = NULL;

    return;
}
// ================== end ne_someip_thread_t object ==============

/** ne_someip_thread_tls_create
 * 将ne_someip_thread对象指针保存到线程TLS
 */
static int32_t ne_someip_thread_tls_create(ne_someip_thread_t *thread);
/** ne_someip_thread_run
 *  thread的run函数，用来运行thread_func
 */
static void* ne_someip_thread_run(void* user_data);


ne_someip_thread_t* ne_someip_thread_new(char* name, ne_someip_thread_user_main_func thread_func, ne_someip_thread_user_free_func free_func, void* user_data)
{
    if (!thread_func) {
        ne_someip_log_error("[Thread] thread_func is null");
        return NULL;
    }

    ne_someip_thread_t* thread = ne_someip_thread_t_new();
    if (!thread) {
        ne_someip_log_error("[Thread] malloc error");
        return NULL;
    }

    thread->user_data = user_data;
    thread->thread_func = thread_func;
    thread->free_func = free_func;
    thread->id = 0;
    thread->is_running = 0;
    thread->looper_run = false;

    if (name && (NE_SOMEIP_THREAD_NAME_SIZE > strlen(name))) {
        strncpy(thread->name, name, strlen(name) + 1);
    }

    return thread;
}

ne_someip_thread_t* ne_someip_thread_new_looper(char* name, ne_someip_thread_user_main_func thread_func, ne_someip_thread_user_free_func free_func, ne_someip_looper_t* looper)
{
    if (NULL == looper) {
        ne_someip_log_error("[Thread] looper is NULL");
        return NULL;
    }
    if (!thread_func) {
        thread_func = ne_someip_looper_run;
    }
    ne_someip_thread_t* thread = ne_someip_thread_new(name, thread_func, free_func, looper);
    if (!thread) {
        ne_someip_log_error("[Thread] thread create error");
        return NULL;
    }
    thread->looper_run = true;
    // thread持有该looper，引用计数+1
    ne_someip_looper_ref(looper);
    ne_someip_log_debug("[Thread] create thread:%p, looper:%p", thread, looper);
    return thread;
}

ne_someip_thread_t* ne_someip_thread_ref(ne_someip_thread_t *thread)
{
    if (!thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return NULL;
    }
    return ne_someip_thread_t_ref(thread);
}

void ne_someip_thread_unref(ne_someip_thread_t *thread)
{
    if (!thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return;
    }
    ne_someip_thread_t_unref(thread);
}

bool ne_someip_thread_is_running(ne_someip_thread_t *thread)
{
    if (!thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return false;
    }
    pthread_mutex_lock(&thread->lock);
    if (0 < thread->is_running) {
	pthread_mutex_unlock(&thread->lock);
        return true;
    }
    pthread_mutex_unlock(&thread->lock);
    return false;
}

int32_t ne_someip_thread_set_name(ne_someip_thread_t *thread, char* name)
{
    if (!thread || false == ne_someip_thread_is_running(thread) || !name || (NE_SOMEIP_THREAD_NAME_SIZE <= strlen(name))) {
        ne_someip_log_error("[Thread] set name faile");
        return -1;
    }

    int32_t ret = -1;

    if (thread->name != name) {
        strncpy(thread->name, name, strlen(name) + 1);
    }
    ret = pthread_setname_np(thread->id, name);

    return ret;
}

char* ne_someip_thread_get_name(ne_someip_thread_t *thread)
{
    if (!thread || false == ne_someip_thread_is_running(thread)) {
        ne_someip_log_error("[Thread] thread is NULL");
        return NULL;
    }
    pthread_getname_np(thread->id, thread->name, NE_SOMEIP_THREAD_NAME_SIZE);
    return thread->name;
}

int32_t ne_someip_thread_start(ne_someip_thread_t *thread)
{
    if (!thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return -1;
    }
    ne_someip_log_debug("[Thread] start run thread:%p", thread);

    if (0 != pthread_create(&(thread->id), &(thread->attr), ne_someip_thread_run, thread)) {
        ne_someip_log_error("[Thread] failure to start thread...");
        return -1;
    }
    // 线程创建，引用+1
    ne_someip_thread_ref(thread);
    thread->is_running = thread->is_running + 1;

    return 0;
}

int32_t ne_someip_thread_stop(ne_someip_thread_t *thread)
{
    int32_t ret;

    ne_someip_thread_notify(thread);
    ret = pthread_join(thread->id, NULL);

    if (ret == 0) {
        ne_someip_log_debug("[Thread] stop run thread:%p", thread);
        thread->is_running = 0;
        thread->id = 0;
        // 线程停止，引用-1
        ne_someip_thread_unref(thread);
    }
    else {
        ne_someip_log_error("[Thread] thread join falid:%d", ret);
    }

    return ret;
}

int32_t ne_someip_thread_tls_create(ne_someip_thread_t *thread)
{
    if (NULL == thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return -1;
    }
    pthread_mutex_lock(&g_tls_mutex);
    if (!g_tls_exist) {
        int32_t ret = pthread_key_create(&g_tls_key_thread, NULL);
        if (0 != ret) {
            pthread_mutex_unlock(&g_tls_mutex);
            ne_someip_log_error("[Thread] key create faild");
            return -1;
        }
        g_tls_exist = 1;
    }
    pthread_mutex_unlock(&g_tls_mutex);
    if (0 != pthread_setspecific(g_tls_key_thread, thread)) {
        ne_someip_log_error("[Thread] set specific faild");
        return -1;
    }
    return 0;
}

void* ne_someip_thread_run(void* user_data)
{
    ne_someip_thread_t* thread = (ne_someip_thread_t*)user_data;

    if (!thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return NULL;
    }

    if (0 != ne_someip_thread_tls_create(thread)) {
        ne_someip_log_error("[Thread] tls create error");
        return NULL;
    }

    ne_someip_log_debug("[Thread] run thread:%p", thread);
    pthread_mutex_lock(&thread->lock);
    thread->is_running += 1;
    pthread_mutex_unlock(&thread->lock);

    ne_someip_thread_set_name(thread, thread->name);

    if (thread->thread_func) {
        thread->thread_func(thread->user_data);
    }

    pthread_mutex_lock(&thread->lock);
    thread->is_running = 0;
    pthread_mutex_unlock(&thread->lock);

    return NULL;
}

ne_someip_thread_t* ne_someip_thread_self()
{
    return pthread_getspecific(g_tls_key_thread);
}

int32_t ne_someip_thread_wait(ne_someip_thread_t* thread)
{
    if (NULL == thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return -1;
    }

    pthread_mutex_lock(&(thread->lock));

    while(!thread->signal_flag) {
        // 防止意外唤醒，需要等待signal_flag设置后再结束wait
        pthread_cond_wait(&(thread->cond), &(thread->lock));
    }

    thread->signal_flag = 0;

    pthread_mutex_unlock(&(thread->lock));
    return 0;
}

int32_t ne_someip_thread_notify(ne_someip_thread_t* thread)
{

    if (NULL == thread) {
        ne_someip_log_error("[Thread] thread is NULL");
        return -1;
    }

    pthread_mutex_lock(&(thread->lock));

    thread->signal_flag = 1;

    int32_t ret = pthread_cond_signal(&(thread->cond));

    pthread_mutex_unlock(&(thread->lock));

    return ret;
}

pthread_t ne_someip_thread_get_tid(ne_someip_thread_t* thread)
{
    if (NULL == thread) {
        return 0;
    }

    return thread->id;
}
