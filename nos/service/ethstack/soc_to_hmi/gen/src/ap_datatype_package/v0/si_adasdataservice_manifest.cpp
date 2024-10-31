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
 * @file si_adasdataservice_0_0_manifest.cpp
 * @brief 
 *
 */


#include "ara/com/internal/manifest_config.h"

static const char* si_adasdataservice_0_0_method_table[] = {
    nullptr
};
static const char* si_adasdataservice_0_0_event_table[] = {
    nullptr
};
static const char* si_adasdataservice_0_0_field_table[] = {
    "ADASdataProperties_Field",
    nullptr
};
static const char* si_adasdataservice_0_0_Application_AdaptiveApplicationSwComponentTypes_SWC_ADCS_ADASdataService_si_adasdataservice_PPort_instance_table[] = {
    "1",
    nullptr
};
static const ara::com::runtime::ComPortMappingConfig si_adasdataservice_0_0_RPort_mapping_config[] =
{
    {nullptr, nullptr}
};

static const ara::com::runtime::ComPortMappingConfig si_adasdataservice_0_0_PPort_mapping_config[] =
{
    {
        "/Application/AdaptiveApplicationSwComponentTypes/SWC_ADCS/ADASdataService",
        si_adasdataservice_0_0_Application_AdaptiveApplicationSwComponentTypes_SWC_ADCS_ADASdataService_si_adasdataservice_PPort_instance_table
    },
    {nullptr, nullptr}
};

static const ara::com::runtime::ComBindingManifestConfig si_adasdataservice_0_0_binding_manifest_config_list[] =
{
    {
        "isomeip",
        false,
        "someip",
        R"({
            "services":
            [
                {
                    "short_name" : "si_adasdataservice_0_0",
                    "service" : 1027,
                    "major_version" : 1,
                    "minor_version" : 0,
                    "events":
                    [
                        {
                            "short_name" : "ADASdataProperties_Field.onChange_ADASdataProperties_Field",
                            "event" : 257,
                            "serializer" : "SOMEIP",
                            "communication_type" : "TCP"
                        }
                    ],
                    "fields":
                    [
                        {
                            "short_name" : "ADASdataProperties_Field",
                            "notifier" : 257,
                            "getter" : 513
                        }
                    ],
                    "methods":
                    [
                        {
                            "short_name":"ADASdataProperties_Field.get_ADASdataProperties_Field",
                            "method": 513,
                            "fire_and_forget" : false,
                            "communication_type" : "TCP"
                        }
                    ],
                    "eventgroups":
                    [
                        {
                            "short_name" : "ADASdataProperties_EG",
                            "eventgroup" : 4353,
                            "events" : [257]
                        }
                    ]
                }
            ],
            "provided_instances" :
            [
                {
                    "service" : 1027,
                    "major_version" : 1,
                    "short_name" : "PSI_ADASdataService_EEI_ADCS",
                    "instance" : 1,
                    "udp_port" : 0,
                    "tcp_port" : 31001,
                    "tcp_reuse" : false,
                    "udp_reuse" : false,
                    "method_attribute" :
                    [
                    ],
                    "event_attribute" :
                    [
                    ],
                    "offer_time_reference" : "ServerSd_PSI_ADASdataService_EEI_ADCS",
                    "ethernet_reference" : "NEP_ADCS",
                    "tcp_tls_flag" : false,
                    "udp_tls_flag" : false,
                    "provided_eventgroups" :
                    [
                        {
                            "eventgroup" : 4353,
                            "threshold" : 0,
                            "subscribe_time_reference" : "ServerEgSd_SomeIpADASdataProperties_EG"
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
                    "sd_time_name" : "ServerSd_PSI_ADASdataService_EEI_ADCS",
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
                    "sd_time_name" : "ServerEgSd_SomeIpADASdataProperties_EG",
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
ara::com::runtime::ComServiceManifestConfig si_adasdataservice_0_0_manifest_config = {
    "si_adasdataservice_0_0",
    si_adasdataservice_0_0_method_table,
    si_adasdataservice_0_0_event_table,
    si_adasdataservice_0_0_field_table,
    si_adasdataservice_0_0_binding_manifest_config_list,
    si_adasdataservice_0_0_PPort_mapping_config,
    si_adasdataservice_0_0_RPort_mapping_config
};
