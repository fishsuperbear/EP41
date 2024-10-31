/**
 * Copyright @ 2019 iAUTO (Shanghai) Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * iAUTO (Shanghai) Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * @file NESomeIPEtsProcess.h
 * @brief
 */

#ifndef TESTS_SOMEIP_SOMEIPETSPROCESS_H_
#define TESTS_SOMEIP_SOMEIPETSPROCESS_H_

#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
// #ifdef __cplusplus
// extern "C" {
// #endif

#include <unistd.h>
#include <stdint.h>
#include <memory>
#include <vector>
#include "ne_someip_define.h"
#include "ne_someip_server_context.h"
#include "ne_someip_config_parse.h"
#include "ne_someip_server_context.h"
#include "ne_someip_client_context.h"

using namespace std;

static std::vector<uint16_t> MethodIDType = {
    0x1F, 0x2F, 0x30, 0x32, 0x23, 0x17, 0x12, 0x34,
    0x0e, 0x36, 0x18, 0x40, 0x08, 0x09, 0x3E, 0x3F,
    0x35, 0x37, 0x0B, 0x0A, 0x19, 0x16, 0x14, 0x15,
    0x13, 0x01, 0x02, 0x03, 0x04, 0x06, 0x05, 0x3A,
    0x3B, 0x3C, 0x3D, 0x41, 0x8001, 0x8003, 0x800B,
    0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x1
};

static ne_someip_client_id_t clientId = 0;
static ne_someip_service_id_t serviceId = 312;
static ne_someip_instance_id_t instanceId = 0xF4;
static ne_someip_major_version_t majorVersion = 1;
static ne_someip_minor_version_t minorVersion = 0;
static ne_someip_instance_id_t instanceId_server = 1;

typedef union _union {
    bool valueb;
    uint8_t value8;
    uint16_t value16;
    uint32_t value32;
    int8_t svalue8;
    int16_t svalue16;
    int32_t svalue32;
    float valuef;
} NESomeIPETSTestUnion;

typedef struct StringUnion {
    char *str_utf8[8];
    NESomeIPETSTestUnion struct_union;
} NESomeIPETSTestStructUnion;

typedef struct String {
    char *str_utf8[8];
} NESomeIPETSTestStruct;

class NESomeIPEtsProcess : public std::enable_shared_from_this<NESomeIPEtsProcess> {
public:
    NESomeIPEtsProcess(ne_someip_server_context_t* server_contest, ne_someip_provided_instance_t* pro_instance);
    virtual ~NESomeIPEtsProcess();
    void setThisPointerToEts();

    ne_someip_return_code_t checkByteOrder(ne_someip_payload_t* Payload);
    ne_someip_return_code_t clientServiceActivate(ne_someip_payload_t* Payload);
    ne_someip_return_code_t clientServiceDeactivate(ne_someip_payload_t* Payload);
    ne_someip_return_code_t clientServiceSubscribeEventgroup(ne_someip_payload_t* Payload);
    ne_someip_return_code_t clientServiceGetLastValueEventTCP(ne_someip_payload_t** Payload);
    ne_someip_return_code_t clientServiceGetLastValueEventUDPUnicast(ne_someip_payload_t** Payload);
    ne_someip_return_code_t clientServiceGetLastValueEventUDPMulticast(ne_someip_payload_t** Payload);
    ne_someip_return_code_t TestEventUINT8(ne_someip_payload_t* Payload);
    ne_someip_return_code_t TestEventUINT8Reliable(ne_someip_payload_t* Payload);
    ne_someip_return_code_t TestEventUINT8Multicast(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoCommonDatatypes(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoENUM(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoFLOAT64(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoInt64(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoINT8(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoStaticUINT8Array(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUINT8(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUINT8Array(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUINT8Array16BitLength(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUINT8Array8BitLength(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUINT8Array2Dim(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUINT8ArrayMinSize(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUNION(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUTF16DYNAMIC(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUTF16FIXED(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUTF8DYNAMIC(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoUTF8FIXED(ne_someip_payload_t* Payload);
    ne_someip_return_code_t suspendInterface(ne_someip_payload_t* Payload);
    ne_someip_return_code_t triggerEventUINT8(ne_someip_payload_t* Payload);
    ne_someip_return_code_t triggerEventUINT8Array(ne_someip_payload_t* Payload);
    ne_someip_return_code_t triggerEventUINT8Reliable(ne_someip_payload_t* Payload);
    ne_someip_return_code_t triggerEventUINT8Multicast(ne_someip_payload_t* Payload);
    ne_someip_return_code_t echoBitfields(ne_someip_payload_t* Payload);
    ne_someip_return_code_t interfaceVersionGetter(ne_someip_header_t** header, ne_someip_payload_t** payload);
    ne_someip_return_code_t testFieldUint8Getter(ne_someip_header_t** header, ne_someip_payload_t** payload);
    ne_someip_return_code_t testFieldUint8Setter(ne_someip_header_t** header, ne_someip_payload_t** payload);
    ne_someip_return_code_t testFieldUint8ArrayGetter(ne_someip_header_t** header, ne_someip_payload_t** payload);
    ne_someip_return_code_t testFieldUint8ArraySetter(ne_someip_header_t** header, ne_someip_payload_t** payload);
    ne_someip_return_code_t testFieldUint8ReliableGetter(ne_someip_header_t** header, ne_someip_payload_t** payload);
    ne_someip_return_code_t testFieldUint8ReliableSetter(ne_someip_header_t** header, ne_someip_payload_t** payload);
    void testResetInterface();

private:
    ne_someip_config_t* g_someip_config;
    ne_someip_server_context_t* g_server_context;
    ne_someip_provided_instance_t* g_pro_instance;
    ne_someip_client_context_t* g_client_context;
    ne_someip_required_service_instance_t* g_req_instance;

    ne_someip_payload_t* m_payload_reliable;
    ne_someip_payload_t* m_payload_multicast;
    ne_someip_payload_t* m_payload_unicast;

    ne_someip_config_t* load_config(const char* someip_config_path);
    ne_someip_required_service_instance_t* create_client_req_instance(const ne_someip_required_service_instance_config_t* inst_config);
    ne_someip_error_code_t start_find_service(const ne_someip_required_service_instance_config_t* inst_config);
    ne_someip_payload_t* create_payload();

};

// #ifdef __cplusplus
// }
// #endif
#endif  // TESTS_SOMEIP_SOMEIPETSPROCESS_H_
/* EOF */