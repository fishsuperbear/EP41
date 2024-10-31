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
#ifndef NE_SOMEIP_SYNC_WAIT_OBJ_H
#define NE_SOMEIP_SYNC_WAIT_OBJ_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <pthread.h>

struct ne_someip_sync_wait_obj;
typedef struct ne_someip_sync_wait_obj ne_someip_sync_wait_obj_t;

// create
ne_someip_sync_wait_obj_t* ne_someip_sync_wait_obj_create();

// destroy
void ne_someip_sync_wait_obj_destroy(ne_someip_sync_wait_obj_t* wait_obj);

// wait
void ne_someip_sync_wait_obj_wait(ne_someip_sync_wait_obj_t* wait_obj);

// time wait
void ne_someip_sync_wait_obj_timedwait(ne_someip_sync_wait_obj_t* wait_obj, uint32_t msec);

// notify
void ne_someip_sync_wait_obj_notify(ne_someip_sync_wait_obj_t* wait_obj);

#ifdef __cplusplus
}
#endif
#endif // NE_SOMEIP_SYNC_WAIT_OBJ_H
/* EOF */