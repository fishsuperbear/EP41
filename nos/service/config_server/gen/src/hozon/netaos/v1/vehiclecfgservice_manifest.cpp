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
 * @file vehiclecfgservice_1_0_manifest.cpp
 * @brief
 *
 */

#include "ara/com/internal/manifest_config.h"

static const char* vehiclecfgservice_1_0_method_table[] = {"VehicleCfgUpdateRes", nullptr};
static const char* vehiclecfgservice_1_0_event_table[] = {"VehicleCfgUpdateEvent", nullptr};
static const char* vehiclecfgservice_1_0_field_table[] = {nullptr};
static const char* vehiclecfgservice_1_0_IAUTOSAR_AASWC_VehicleCfgService_Server_SOC_VehicleCfgService_1_PPort_vehiclecfgservice_PPort_instance_table[] = {"1", nullptr};
static const ara::com::runtime::ComPortMappingConfig vehiclecfgservice_1_0_RPort_mapping_config[] = {{nullptr, nullptr}};

static const ara::com::runtime::ComPortMappingConfig vehiclecfgservice_1_0_PPort_mapping_config[] = {
    {"/IAUTOSAR/AASWC_VehicleCfgService_Server_SOC/VehicleCfgService_1_PPort",
     vehiclecfgservice_1_0_IAUTOSAR_AASWC_VehicleCfgService_Server_SOC_VehicleCfgService_1_PPort_vehiclecfgservice_PPort_instance_table},
    {nullptr, nullptr}};

static const ara::com::runtime::ComBindingManifestConfig vehiclecfgservice_1_0_binding_manifest_config_list[] = {{"isomeip", false, "someip",
                                                                                                                  R"({
            "services":
            [
                {
                    "short_name" : "vehiclecfgservice_1_0",
                    "service" : 25006,
                    "major_version" : 1,
                    "minor_version" : 0,
                    "events":
                    [
                        {
                            "short_name" : "VehicleCfgUpdateEvent",
                            "event" : 8,
                            "communication_type" : "UDP"
                        }
                    ],
                    "fields":
                    [
                    ],
                    "methods":
                    [
                        {
                            "short_name":"VehicleCfgUpdateRes",
                            "method": 519,
                            "fire_and_forget" : false,
                            "communication_type" : "UDP"
                        }
                    ],
                    "eventgroups":
                    [
                        {
                            "short_name" : "VehicleCfgService_259_EventGroup",
                            "eventgroup" : 259,
                            "events" : [8]
                        }
                    ]
                }
            ],
            "provided_instances" :
            [
                {
                    "service" : 25006,
                    "major_version" : 1,
                    "short_name" : "VehicleCfgService_SOC_SomeipPIns_1",
                    "instance" : 1,
                    "udp_port" : 10004,
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
                            "eventgroup" : 259,
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
        })"},
#if 0
    {
        "dds",
        false,
        "dds",
        R"(json)"
    },
#endif
                                                                                                                 {nullptr, false, nullptr, nullptr}};
ara::com::runtime::ComServiceManifestConfig vehiclecfgservice_1_0_manifest_config = {"vehiclecfgservice_1_0",
                                                                                     vehiclecfgservice_1_0_method_table,
                                                                                     vehiclecfgservice_1_0_event_table,
                                                                                     vehiclecfgservice_1_0_field_table,
                                                                                     vehiclecfgservice_1_0_binding_manifest_config_list,
                                                                                     vehiclecfgservice_1_0_PPort_mapping_config,
                                                                                     vehiclecfgservice_1_0_RPort_mapping_config};