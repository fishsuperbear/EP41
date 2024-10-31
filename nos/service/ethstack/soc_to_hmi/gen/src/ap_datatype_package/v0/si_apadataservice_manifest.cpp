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
 * @file si_apadataservice_0_0_manifest.cpp
 * @brief 
 *
 */


#include "ara/com/internal/manifest_config.h"

static const char* si_apadataservice_0_0_method_table[] = {
    nullptr
};
static const char* si_apadataservice_0_0_event_table[] = {
    nullptr
};
static const char* si_apadataservice_0_0_field_table[] = {
    "APAdataProperties_Field",
    "HPPInfo_Field",
    "HPPLocationProperties_Field",
    "HPPMapObjectDisplay_Field",
    "HPPdataProperties_Field",
    "InsInfoProperties_Field",
    "NNSInfoProperties_Field",
    nullptr
};
static const char* si_apadataservice_0_0_Application_AdaptiveApplicationSwComponentTypes_SWC_ADCS_APAdataService_si_apadataservice_PPort_instance_table[] = {
    "1",
    nullptr
};
static const ara::com::runtime::ComPortMappingConfig si_apadataservice_0_0_RPort_mapping_config[] =
{
    {nullptr, nullptr}
};

static const ara::com::runtime::ComPortMappingConfig si_apadataservice_0_0_PPort_mapping_config[] =
{
    {
        "/Application/AdaptiveApplicationSwComponentTypes/SWC_ADCS/APAdataService",
        si_apadataservice_0_0_Application_AdaptiveApplicationSwComponentTypes_SWC_ADCS_APAdataService_si_apadataservice_PPort_instance_table
    },
    {nullptr, nullptr}
};

static const ara::com::runtime::ComBindingManifestConfig si_apadataservice_0_0_binding_manifest_config_list[] =
{
    {
        "isomeip",
        false,
        "someip",
        R"({
            "services":
            [
                {
                    "short_name" : "si_apadataservice_0_0",
                    "service" : 1028,
                    "major_version" : 1,
                    "minor_version" : 0,
                    "events":
                    [
                        {
                            "short_name" : "APAdataProperties_Field.onChange_APAdataProperties_Field",
                            "event" : 257,
                            "serializer" : "SOMEIP",
                            "communication_type" : "TCP"
                        },
                        {
                            "short_name" : "HPPLocationProperties_Field.onChange_HPPLocationProperties_Field",
                            "event" : 259,
                            "serializer" : "SOMEIP",
                            "communication_type" : "TCP"
                        },
                        {
                            "short_name" : "HPPMapObjectDisplay_Field.onChange_HPPMapObjectDisplay_Field",
                            "event" : 260,
                            "serializer" : "SOMEIP",
                            "communication_type" : "TCP"
                        },
                        {
                            "short_name" : "HPPdataProperties_Field.onChange_HPPdataProperties_Field",
                            "event" : 258,
                            "serializer" : "SOMEIP",
                            "communication_type" : "TCP"
                        }
                    ],
                    "fields":
                    [
                        {
                            "short_name" : "APAdataProperties_Field",
                            "notifier" : 257,
                            "getter" : 513
                        },
                        {
                            "short_name" : "HPPInfo_Field",
                            "setter" : 261
                        },
                        {
                            "short_name" : "HPPLocationProperties_Field",
                            "notifier" : 259
                        },
                        {
                            "short_name" : "HPPMapObjectDisplay_Field",
                            "notifier" : 260
                        },
                        {
                            "short_name" : "HPPdataProperties_Field",
                            "notifier" : 258,
                            "getter" : 514
                        },
                        {
                            "short_name" : "InsInfoProperties_Field",
                            "setter" : 263
                        },
                        {
                            "short_name" : "NNSInfoProperties_Field",
                            "setter" : 262
                        }
                    ],
                    "methods":
                    [
                        {
                            "short_name":"APAdataProperties_Field.get_APAdataProperties_Field",
                            "method": 513,
                            "fire_and_forget" : false,
                            "communication_type" : "TCP"
                        },
                        {
                            "short_name":"HPPInfo_Field.set_HPPInfo_Field",
                            "method": 261,
                            "fire_and_forget" : false,
                            "communication_type" : "TCP"
                        },
                        {
                            "short_name":"HPPdataProperties_Field.get_HPPdataProperties_Field",
                            "method": 514,
                            "fire_and_forget" : false,
                            "communication_type" : "TCP"
                        },
                        {
                            "short_name":"InsInfoProperties_Field.set_InsInfoProperties_Field",
                            "method": 263,
                            "fire_and_forget" : false,
                            "communication_type" : "TCP"
                        },
                        {
                            "short_name":"NNSInfoProperties_Field.set_NNSInfoProperties_Field",
                            "method": 262,
                            "fire_and_forget" : false,
                            "communication_type" : "TCP"
                        }
                    ],
                    "eventgroups":
                    [
                        {
                            "short_name" : "APAdataProperties_EG",
                            "eventgroup" : 4353,
                            "events" : [257]
                        },
                        {
                            "short_name" : "HPPdataProperties_EG",
                            "eventgroup" : 4354,
                            "events" : [259,260,258]
                        }
                    ]
                }
            ],
            "provided_instances" :
            [
                {
                    "service" : 1028,
                    "major_version" : 1,
                    "short_name" : "PSI_APAdataService_EEI_ADCS",
                    "instance" : 1,
                    "udp_port" : 0,
                    "tcp_port" : 31002,
                    "tcp_reuse" : false,
                    "udp_reuse" : false,
                    "method_attribute" :
                    [
                    ],
                    "event_attribute" :
                    [
                    ],
                    "offer_time_reference" : "ServerSd_PSI_APAdataService_EEI_ADCS",
                    "ethernet_reference" : "NEP_ADCS",
                    "tcp_tls_flag" : false,
                    "udp_tls_flag" : false,
                    "provided_eventgroups" :
                    [
                        {
                            "eventgroup" : 4353,
                            "threshold" : 0,
                            "subscribe_time_reference" : "ServerEgSd_SomeIpAPAdataProperties_EG"
                        },
                        {
                            "eventgroup" : 4354,
                            "threshold" : 0,
                            "subscribe_time_reference" : "ServerEgSd_SomeIpHPPdataProperties_EG"
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
                    "sd_time_name" : "ServerSd_PSI_APAdataService_EEI_ADCS",
                    "initial_delay_min" : 10,
                    "initial_delay_max" : 50,
                    "repetitions_base_delay" : 50,
                    "repetitions_max" : 3,
                    "ttl" : 3,
                    "cyclic_offer_delay" : 1000,
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
                {
                    "sd_time_name" : "ServerEgSd_SomeIpAPAdataProperties_EG",
                    "initial_delay_min" : 10,
                    "initial_delay_max" : 50
                },
                {
                    "sd_time_name" : "ServerEgSd_SomeIpHPPdataProperties_EG",
                    "initial_delay_min" : 10,
                    "initial_delay_max" : 50
                }
            ],
            "network_endpoints" :
            [{
                    "network_id" : "NEP_ADCS",
                    "ip_addr" : "172.16.1.40",
                    "ip_addr_type" : 4,
                    "subnet_mask" : "255.255.255.0",
                    "multicast_ip_addr" : "239.127.3.1",
                    "multicast_port" : 30490
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
ara::com::runtime::ComServiceManifestConfig si_apadataservice_0_0_manifest_config = {
    "si_apadataservice_0_0",
    si_apadataservice_0_0_method_table,
    si_apadataservice_0_0_event_table,
    si_apadataservice_0_0_field_table,
    si_apadataservice_0_0_binding_manifest_config_list,
    si_apadataservice_0_0_PPort_mapping_config,
    si_apadataservice_0_0_RPort_mapping_config
};
