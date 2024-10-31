/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file socudsservice_1_0_manifest.cpp
 * @brief 
 *
 */


#include "ara/com/internal/manifest_config.h"

static const char* socudsservice_1_0_method_table[] = {
    "McuUdsRes",
    nullptr
};
static const char* socudsservice_1_0_event_table[] = {
    "SocUdsReq",
    nullptr
};
static const char* socudsservice_1_0_field_table[] = {
    nullptr
};
static const char* socudsservice_1_0_IAUTOSAR_AASWC_SoCUdsService_Server_SOC_SoCUdsService_1_PPort_socudsservice_PPort_instance_table[] = {
    "1",
    nullptr
};
static const ara::com::runtime::ComPortMappingConfig socudsservice_1_0_RPort_mapping_config[] =
{
    {nullptr, nullptr}
};

static const ara::com::runtime::ComPortMappingConfig socudsservice_1_0_PPort_mapping_config[] =
{
    {
        "/IAUTOSAR/AASWC_SoCUdsService_Server_SOC/SoCUdsService_1_PPort",
        socudsservice_1_0_IAUTOSAR_AASWC_SoCUdsService_Server_SOC_SoCUdsService_1_PPort_socudsservice_PPort_instance_table
    },
    {nullptr, nullptr}
};

static const ara::com::runtime::ComBindingManifestConfig socudsservice_1_0_binding_manifest_config_list[] =
{
    {
        "isomeip",
        false,
        "someip",
        R"({
            "services":
            [
                {
                    "short_name" : "socudsservice_1_0",
                    "service" : 25018,
                    "major_version" : 1,
                    "minor_version" : 0,
                    "events":
                    [
                        {
                            "short_name" : "SocUdsReq",
                            "event" : 36,
                            "communication_type" : "UDP"
                        }
                    ],
                    "fields":
                    [
                    ],
                    "methods":
                    [
                        {
                            "short_name":"McuUdsRes",
                            "method": 771,
                            "fire_and_forget" : false,
                            "communication_type" : "UDP"
                        }
                    ],
                    "eventgroups":
                    [
                        {
                            "short_name" : "SoCUdsService_264_EventGroup",
                            "eventgroup" : 264,
                            "events" : [36]
                        }
                    ]
                }
            ],
            "provided_instances" :
            [
                {
                    "service" : 25018,
                    "major_version" : 1,
                    "short_name" : "SoCUdsService_SOC_SomeipPIns_1",
                    "instance" : 1,
                    "udp_port" : 10010,
                    "tcp_port" : 0,
                    "tcp_reuse" : false,
                    "udp_reuse" : false,
                    "method_attribute" :
                    [
                    ],
                    "event_attribute" :
                    [
                    ],
                    "offer_time_reference" : "ServiceProvidedSomeipSD",
                    "ethernet_reference" : "NEP_SOC_1",
                    "tcp_tls_flag" : false,
                    "udp_tls_flag" : false,
                    "provided_eventgroups" :
                    [
                        {
                            "eventgroup" : 264,
                            "multicast_addr_ip4" : "239.192.255.123",
                            "multicast_port" : 30491,
                            "threshold" : 0
                        }
                    ]
                }
            ],
            "required_instances" :
            [
            ],
            "sd_server_offer_times" :
            [
                {
                    "sd_time_name" : "ServiceProvidedSomeipSD",
                    "initial_delay_min" : 10,
                    "initial_delay_max" : 100,
                    "repetitions_base_delay" : 200,
                    "repetitions_max" : 3,
                    "ttl" : 10,
                    "cyclic_offer_delay" : 2000,
                    "request_response_delay_min" : 10,
                    "request_response_delay_max" : 50
                }
            ],
            "sd_client_find_times" :
            [
            ],
            "sd_client_subscribe_times" :
            [
            ],
            "sd_server_subscribe_times" :
            [
            ],
            "network_endpoints" :
            [{
                    "network_id" : "NEP_SOC_1",
                    "ip_addr" : "172.16.90.11",
                    "ip_addr_type" : 4,
                    "subnet_mask" : "255.255.255.0",
                    "multicast_ip_addr" : "239.192.255.123",
                    "multicast_port" : 30491
                }
            ]
        })"
    },
#if 0
    {
        "dds",
        false,
        "dds",
        R"(json)"
    },
#endif
    {nullptr, false, nullptr, nullptr}
};
ara::com::runtime::ComServiceManifestConfig socudsservice_1_0_manifest_config = {
    "socudsservice_1_0",
    socudsservice_1_0_method_table,
    socudsservice_1_0_event_table,
    socudsservice_1_0_field_table,
    socudsservice_1_0_binding_manifest_config_list,
    socudsservice_1_0_PPort_mapping_config,
    socudsservice_1_0_RPort_mapping_config
};
