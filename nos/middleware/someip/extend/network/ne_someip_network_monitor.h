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
#ifndef SRC_EXTEND_NETLINK_NE_SOMEIP_NETWORK_MONITOR_H
#define SRC_EXTEND_NETLINK_NE_SOMEIP_NETWORK_MONITOR_H

#include <stdbool.h>
#include "ne_someip_list.h"

#define NE_SOMEIP_NETWORK_IFNAME_LEN 16
#define NE_SOMEIP_NETWORK_SELECT_TIMEOUT 1
#define NE_SOMEIP_NETWORK_BUFF_SIZE 1024
#define NE_SOMEIP_NETLINK_BUFF_SIZE 2048
#define NE_SOMEIP_NETWORK_SLEEP_SECONDS 1

typedef void (*ne_someip_network_status_handler)(const char* ifname, uint32_t ip_addr,
    bool is_enabled, void* user_data);

typedef struct ne_someip_network_handler
{
    ne_someip_network_status_handler handler;
    void* user_data;
} ne_someip_network_handler_t;

typedef struct ne_someip_network_ifname_info
{
    uint32_t ip_addr;
    bool link_status;
    bool ip_status;
    bool link_available;
} ne_someip_network_ifname_info_t;

typedef struct ne_someip_network_ip_info
{
    char ifname[NE_SOMEIP_NETWORK_IFNAME_LEN];
    bool link_status;
    bool ip_status;
    bool link_available;
} ne_someip_network_ip_info_t;

typedef struct ne_someip_network_monitor ne_someip_network_monitor_t;

bool ne_someip_network_monitor_init();
void ne_someip_network_monitor_deinit();
bool ne_someip_network_monitor_start();
bool ne_someip_network_monitor_stop();

bool ne_someip_network_monitor_register_callback_by_ifname(const char* ifname,
    ne_someip_network_status_handler handler, void* user_data);
bool ne_someip_network_monitor_register_callback_by_ip(uint32_t ip_addr,
    ne_someip_network_status_handler handler, void* user_data);
bool ne_someip_network_monitor_unregister_callback(void* user_data);

/******************free func************************/
void ne_someip_network_ifname_info_free(void* data);
void ne_someip_network_ip_info_free(void* data);
void ne_someip_network_listener_handler_free(void* data);
void ne_someip_network_ip_free(void* data);
void ne_someip_network_ifname_free(void* data);
void ne_someip_network_listener_handler_list_free(void* data);

#endif // SRC_EXTEND_NETLINK_NE_SOMEIP_NETWORK_MONITOR_H
/* EOF */
