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
#include <pthread.h>
#include <stdlib.h>

#include "ne_someip_sync_obj.h"

// define someip syncobj struct
struct ne_someip_sync_obj
{
    pthread_mutex_t mutex;      /* mutex */
    pthread_mutexattr_t attr;   /* mutex attr */
};

ne_someip_sync_obj_t* ne_someip_sync_obj_create()
{
    ne_someip_sync_obj_t* sync_obj = malloc(sizeof(ne_someip_sync_obj_t));
    if (NULL == sync_obj) {
        return NULL;
    }

    pthread_mutexattr_init(&(sync_obj->attr));
    pthread_mutexattr_settype(&(sync_obj->attr), PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&(sync_obj->mutex), &(sync_obj->attr));

    return sync_obj;
}

void ne_someip_sync_obj_destroy(ne_someip_sync_obj_t* sync_obj)
{
    if (NULL != sync_obj) {
        pthread_mutexattr_destroy(&(sync_obj->attr));
        pthread_mutex_destroy(&(sync_obj->mutex));

        free(sync_obj);
    }
}

void ne_someip_sync_obj_sync_start(ne_someip_sync_obj_t* sync_obj)
{
    if (NULL != sync_obj) {
        pthread_mutex_lock(&(sync_obj->mutex));
    }
}

int ne_someip_sync_obj_try_sync_start(ne_someip_sync_obj_t* sync_obj)
{
    if (NULL == sync_obj) {
        return -1;
    }

    return pthread_mutex_trylock(&(sync_obj->mutex));
}

void ne_someip_sync_obj_sync_end(ne_someip_sync_obj_t* sync_obj)
{
    if (NULL != sync_obj) {
        pthread_mutex_unlock(&(sync_obj->mutex));
    }
}
/* EOF */
