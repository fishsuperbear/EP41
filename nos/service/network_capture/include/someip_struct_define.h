/*
  * Copyright (C) 2020 HAW Hamburg
  *
  * This file is subject to the terms and conditions of the GNU Lesser
  * General Public License v2.1. See the file LICENSE in the top level
  * directory for more details.
  */

/*
 * SOMEIP Header (R19-11)
 *
 *     0                   1                   2                   3
 *     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |               Message ID (Service ID/Method ID)               |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                           Length                              |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |               Request ID (Client ID/Session ID)               |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    | Protocol Ver  | Interface Ver | Message Type  |  Return Code  |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                            Payload                            |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */

#ifndef SOMEIP_STRUCT_DEFINE_H
#define SOMEIP_STRUCT_DEFINE_H
#pragma once

#include <stdint.h>
#include <string>
#include <vector>
namespace hozon {
namespace netaos {
namespace network_capture {

#define SOMEIP_HDR_LENGTH (8U)
#define SOMEIP_FULL_HDR_SIZE (16U)
#define MessageID(a, b) ((uint32_t)a << 16 | (uint32_t)b)

typedef struct {
    uint16_t service_id;
    uint16_t method_id;
} someip_message_id_t;

typedef struct {
    uint16_t client_id;
    uint16_t session_id;
} someip_request_id_t;

typedef struct __attribute__((packed)) {
    someip_message_id_t message_id;
    uint32_t length;
    someip_request_id_t request_id;
    uint8_t protocol_version;
    uint8_t interface_version;
    uint8_t msg_type;
    uint8_t return_code;
} someip_hdr_t;

typedef struct __attribute__((packed)) {
    someip_hdr_t someip_hdr;
    uint32_t data_len;
} someip_message_t;

typedef struct __attribute__((packed)) {
    uint32_t offset;
    bool is_more;
    uint32_t length;
    char* payload;
} someip_tp_segment_t;

struct key_value {
    int key;
    char value[64];
};

struct raw_someip_message {
    std::string topic;
    std::vector<char> msg;
};

#define RECORD_DEBUG false

#if RECORD_DEBUG == true
    #define debug_printf(format, ...) printf(format, ##__VA_ARGS__)
#else
    #define debug_printf(format, ...)
#endif

#endif
}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
