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
 * @file socdataservice_1_0_manifest.cpp
 * @brief 
 *
 */


#include "ara/com/internal/manifest_config.h"

static const char* socdataservice_1_0_method_table[] = {
    nullptr
};
static const char* socdataservice_1_0_event_table[] = {
    "TrajData",
    "PoseData",
    "SnsrFsnLaneDate",
    "SnsrFsnObj",
    "ApaStateMachine",
    "AlgEgoToMCU",
    "APAToMCUChassis",
    "EgoToMCUChassis",
    nullptr
};
static const char* socdataservice_1_0_field_table[] = {
    nullptr
};
static const char* socdataservice_1_0_IAUTOSAR_AASWC_SocDataService_Server_SOC_SocDataService_1_PPort_socdataservice_PPort_instance_table[] = {
    "1",
    nullptr
};
static const ara::com::runtime::ComPortMappingConfig socdataservice_1_0_RPort_mapping_config[] =
{
    {nullptr, nullptr}
};

static const ara::com::runtime::ComPortMappingConfig socdataservice_1_0_PPort_mapping_config[] =
{
    {
        "/IAUTOSAR/AASWC_SocDataService_Server_SOC/SocDataService_1_PPort",
        socdataservice_1_0_IAUTOSAR_AASWC_SocDataService_Server_SOC_SocDataService_1_PPort_socdataservice_PPort_instance_table
    },
    {nullptr, nullptr}
};

static const ara::com::runtime::ComBindingManifestConfig socdataservice_1_0_binding_manifest_config_list[] =
{
    {
        "isomeip",
        false,
        "someip",
        R"({
            "services":
            [
                {
                    "short_name" : "socdataservice_1_0",
                    "service" : 25000,
                    "major_version" : 1,
                    "minor_version" : 0,
                    "events":
                    [
                        {
                            "short_name" : "TrajData",
                            "event" : 5,
                            "segment_length" : 1404,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "PoseData",
                            "event" : 6,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "SnsrFsnLaneDate",
                            "event" : 7,
                            "segment_length" : 1404,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "SnsrFsnObj",
                            "event" : 8,
                            "segment_length" : 1404,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "ApaStateMachine",
                            "event" : 11,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "AlgEgoToMCU",
                            "event" : 12,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "APAToMCUChassis",
                            "event" : 13,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "EgoToMCUChassis",
                            "event" : 14,
                            "communication_type" : "UDP"
                        }
                    ],
                    "fields":
                    [
                    ],
                    "methods":
                    [
                    ],
                    "eventgroups":
                    [
                        {
                            "short_name" : "SocDataService_257_EventGroup",
                            "eventgroup" : 257,
                            "events" : [5,6,7,8,11,12,13,14]
                        }
                    ]
                }
            ],
            "provided_instances" :
            [
                {
                    "service" : 25000,
                    "major_version" : 1,
                    "short_name" : "SocDataService_SOC_SomeipPIns_1",
                    "instance" : 1,
                    "udp_port" : 10002,
                    "tcp_port" : 0,
                    "tcp_reuse" : false,
                    "udp_reuse" : false,
                    "tp_separation_time_usec" : 1000,
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
                            "eventgroup" : 257,
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
ara::com::runtime::ComServiceManifestConfig socdataservice_1_0_manifest_config = {
    "socdataservice_1_0",
    socdataservice_1_0_method_table,
    socdataservice_1_0_event_table,
    socdataservice_1_0_field_table,
    socdataservice_1_0_binding_manifest_config_list,
    socdataservice_1_0_PPort_mapping_config,
    socdataservice_1_0_RPort_mapping_config
};
