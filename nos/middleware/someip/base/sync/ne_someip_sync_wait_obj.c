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
#include <string.h>
#include <stdlib.h>

#include "ne_someip_sync_wait_obj.h"
#include "ne_someip_looper.h"

struct ne_someip_sync_wait_obj
{
    pthread_mutex_t mutex;       /* mutex */
    pthread_mutexattr_t attr;    /* mutex attr */
    pthread_cond_t cond;         /* cond */
    pthread_condattr_t condattr; /* condattr */
    int signal_flag;
};

ne_someip_sync_wait_obj_t* ne_someip_sync_wait_obj_create()
{
    ne_someip_sync_wait_obj_t* wait_obj = malloc(sizeof(ne_someip_sync_wait_obj_t));
    if (NULL == wait_obj) {
        return NULL;
    }
    memset(wait_obj, 0, sizeof(ne_someip_sync_wait_obj_t));
    wait_obj->signal_flag = 0;

    pthread_mutex_init(&(wait_obj->mutex), NULL);

    pthread_condattr_init(&(wait_obj->condattr));
    pthread_condattr_setclock(&wait_obj->condattr, 1);

    pthread_cond_init(&wait_obj->cond, &wait_obj->condattr);

    return wait_obj;
}

void ne_someip_sync_wait_obj_destroy(ne_someip_sync_wait_obj_t* wait_obj)
{
    if (NULL != wait_obj) {
        pthread_mutex_destroy(&(wait_obj->mutex));
        pthread_cond_destroy(&(wait_obj->cond));
        pthread_condattr_destroy(&(wait_obj->condattr));

        free(wait_obj);
        wait_obj = NULL;
    }
}

void ne_someip_sync_wait_obj_wait(ne_someip_sync_wait_obj_t* wait_obj)
{
    if (NULL == wait_obj) {
        return;
    }
    pthread_mutex_lock(&(wait_obj->mutex));
    while(!wait_obj->signal_flag) {
        // 防止意外唤醒，需要等待signal_flag设置后再结束wait
        pthread_cond_wait(&(wait_obj->cond), &(wait_obj->mutex));
    }

    wait_obj->signal_flag = 0;

    pthread_mutex_unlock(&(wait_obj->mutex));
}

void ne_someip_sync_wait_obj_timedwait(ne_someip_sync_wait_obj_t* wait_obj, uint32_t msec)
{
    if (NULL == wait_obj) {
        return;
    }
    pthread_mutex_lock(&(wait_obj->mutex));

    struct timespec nptime;
    ne_someip_looper_time_get_timespec(&nptime, msec);

    if (!wait_obj->signal_flag) {
        pthread_cond_timedwait(&(wait_obj->cond), &(wait_obj->mutex), &nptime);
    }

    wait_obj->signal_flag = 0;

    pthread_mutex_unlock(&(wait_obj->mutex));
}

void ne_someip_sync_wait_obj_notify(ne_someip_sync_wait_obj_t* wait_obj)
{
    if (NULL == wait_obj) {
        return;
    }
    pthread_mutex_lock(&(wait_obj->mutex));
    wait_obj->signal_flag = 1;
    pthread_cond_signal(&(wait_obj->cond));
    pthread_mutex_unlock(&(wait_obj->mutex));
}

/* EOF */
