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
 * @file mcudataservice_1_0_manifest.cpp
 * @brief 
 *
 */


#include "ara/com/internal/manifest_config.h"

static const char* mcudataservice_1_0_method_table[] = {
    nullptr
};
static const char* mcudataservice_1_0_event_table[] = {
    "MbdDebugData",
    "AlgImuInsInfo",
    "AlgGNSSPosInfo",
    "AlgChassisInfo",
    "AlgPNCControl",
    "AlgMcuToEgo",
    "AlgUssRawdata",
    nullptr
};
static const char* mcudataservice_1_0_field_table[] = {
    nullptr
};
static const char* mcudataservice_1_0_IAUTOSAR_AASWC_McuDataService_Client_SOC_McuDataService_1_RPort_mcudataservice_RPort_instance_table[] = {
    "1",
    nullptr
};
static const ara::com::runtime::ComPortMappingConfig mcudataservice_1_0_RPort_mapping_config[] =
{
    {
        "/IAUTOSAR/AASWC_McuDataService_Client_SOC/McuDataService_1_RPort",
        mcudataservice_1_0_IAUTOSAR_AASWC_McuDataService_Client_SOC_McuDataService_1_RPort_mcudataservice_RPort_instance_table
    },
    {nullptr, nullptr}
};

static const ara::com::runtime::ComPortMappingConfig mcudataservice_1_0_PPort_mapping_config[] =
{
    {nullptr, nullptr}
};

static const ara::com::runtime::ComBindingManifestConfig mcudataservice_1_0_binding_manifest_config_list[] =
{
    {
        "isomeip",
        false,
        "someip",
        R"({
            "services":
            [
                {
                    "short_name" : "mcudataservice_1_0",
                    "service" : 25002,
                    "major_version" : 1,
                    "minor_version" : 0,
                    "events":
                    [
                        {
                            "short_name" : "MbdDebugData",
                            "event" : 5,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "AlgImuInsInfo",
                            "event" : 6,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "AlgGNSSPosInfo",
                            "event" : 7,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "AlgChassisInfo",
                            "event" : 8,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "AlgPNCControl",
                            "event" : 9,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "AlgMcuToEgo",
                            "event" : 10,
                            "communication_type" : "UDP"
                        },
                        {
                            "short_name" : "AlgUssRawdata",
                            "event" : 11,
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
                            "short_name" : "McuDataService_258_EventGroup",
                            "eventgroup" : 258,
                            "events" : [5,6,7,8,9,10,11]
                        }
                    ]
                }
            ],
            "provided_instances" :
            [
            ],
            "required_instances" :
            [
                {
                    "service" : 25002,
                    "major_version" : 1,
                    "short_name" : "McuDataService_SOC_SomeipRIns_1",
                    "instance" : 1,
                    "udp_port" : 10001,
                    "tcp_port" : 0,
                    "tp_separation_time_usec" : 1000,
                    "tcp_reuse" : false,
                    "udp_reuse" : false,
                    "method_attribute" :
                    [
                    ],
                    "find_time_reference" : "ServiceRequiredSomeipSD",
                    "ethernet_reference" :  "NEP_SOC_1",
                    "tcp_tls_flag" : false,
                    "udp_tls_flag" : false,
                    "required_eventgroups" :
                    [
                        {
                            "eventgroup" : 258
                        }
                    ]
                }
            ],
            "sd_server_offer_times" :
            [
            ],
            "sd_client_find_times" :
            [
                {
                    "sd_time_name" : "ServiceRequiredSomeipSD",
                    "initial_delay_min" : 10,
                    "initial_delay_max" : 100,
                    "repetitions_base_delay" : 200,
                    "repetitions_max" : 3,
                    "ttl" : 10
                }
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
ara::com::runtime::ComServiceManifestConfig mcudataservice_1_0_manifest_config = {
    "mcudataservice_1_0",
    mcudataservice_1_0_method_table,
    mcudataservice_1_0_event_table,
    mcudataservice_1_0_field_table,
    mcudataservice_1_0_binding_manifest_config_list,
    mcudataservice_1_0_PPort_mapping_config,
    mcudataservice_1_0_RPort_mapping_config
};
