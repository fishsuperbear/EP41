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
#ifndef NE_SOMEIP_SYNC_OBJ_H
#define NE_SOMEIP_SYNC_OBJ_H

#ifdef __cplusplus
extern "C" {
#endif

struct ne_someip_sync_obj;
typedef struct ne_someip_sync_obj ne_someip_sync_obj_t;

// create
ne_someip_sync_obj_t* ne_someip_sync_obj_create();

// sync start
void ne_someip_sync_obj_sync_start(ne_someip_sync_obj_t* sync_obj);

// try sync start
int ne_someip_sync_obj_try_sync_start(ne_someip_sync_obj_t* sync_obj);

// sync end
void ne_someip_sync_obj_sync_end(ne_someip_sync_obj_t* sync_obj);

// destroy
void ne_someip_sync_obj_destroy(ne_someip_sync_obj_t* sync_obj);

#ifdef __cplusplus
}
#endif
#endif // NE_SOMEIP_SYNC_OBJ_H
/* EOF */