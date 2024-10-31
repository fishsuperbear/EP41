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

#ifndef MANAGER_NE_SOMEIP_INIT_H
#define MANAGER_NE_SOMEIP_INIT_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ne_someip_daemon ne_someip_daemon_t;
/**
 *@brief init daemon
 */
ne_someip_daemon_t* ne_someip_daemon_init();

/**
 *@brief deinit daemon
 */
void ne_someip_daemon_deinit();
#ifdef __cplusplus
}
#endif
#endif
/* EOF */