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
#ifndef MANAGER_NE_SOMEIP_DAEMON_INTERNAL_H
#define MANAGER_NE_SOMEIP_DAEMON_INTERNAL_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "ne_someip_endpoint_unix.h"
/**
 *@brief get the unix endpoint
 */
ne_someip_endpoint_unix_t* ne_someip_daemon_get_unix_endpoint();

/**
 * @brief get the client id.
 */
bool ne_someip_daemon_get_client_id(const ne_someip_endpoint_unix_addr_t* unix_addr,
    ne_someip_client_id_t* client_id, ne_someip_client_id_t* client_id_min, ne_someip_client_id_t* client_id_max);

/**
 * @brief remove the client id.
 */
void ne_someip_daemon_refresh_client_id(const ne_someip_endpoint_unix_addr_t* unix_addr);

#ifdef __cplusplus
}
#endif
#endif
/* EOF */
