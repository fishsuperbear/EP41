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

#ifndef NET_SOMEIP_H
#define NET_SOMEIP_H
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

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

static const struct key_value message_type_values[] = {{0x00, "REQUEST"},    {0x01, "REQUEST_NO_RETURN"},    {0x02, "NOTIFICATION"},    {0x80, "RESPONSE"},    {0x81, "ERROR"},
                                                       {0x20, "TP_REQUEST"}, {0x21, "TP_REQUEST_NO_RETURN"}, {0x22, "TP_NOTIFICATION"}, {0xa0, "TP_RESPONSE"}, {0xa1, "TP_ERROR"},
                                                       {0, "NULL"}};

static const struct key_value return_code_values[] = {{0x00, "E_OK"},
                                                      {0x01, "E_NOT_OK"},
                                                      {0x02, "E_UNKNOWN_SERVICE"},
                                                      {0x03, "E_UNKNOWN_METHOD"},
                                                      {0x04, "E_NOT_READY"},
                                                      {0x05, "E_NOT_REACHABLE"},
                                                      {0x06, "E_TIMEOUT"},
                                                      {0x07, "E_WRONG_PROTOCOL_VERSION"},
                                                      {0x08, "E_WRONG_INTERFACE_VERSION"},
                                                      {0x09, "E_MALFORMED_MESSAGE"},
                                                      {0x0a, "E_WRONG_MESSAGE_TYPE"},
                                                      {0x0b, "E_E2E_REPEATED"},
                                                      {0x0c, "E_E2E_WRONG_SEQUENCE"},
                                                      {0x0d, "E_E2E"},
                                                      {0x0e, "E_E2E_NOT_AVAILABLE"},
                                                      {0x0f, "E_E2E_NO_NEW_DATA"},
                                                      {0, "NULL"}};

static const struct key_value service_id_values[] = {{0x61A8, "SocDataService"},
                                                     {0x61AA, "McuDataService"},
                                                     //  {0x61AC, "SocFaultService"},
                                                     //  {0x61AE, "VehicleCfgService"},
                                                     //  {0x61B0, "SocPowerService"},
                                                     //  {0x61B2, "McuFaultService"},
                                                     //  {0x61B4, "SocSentryService"},
                                                     //  {0x61B6, "McuSentryService"},
                                                     //  {0x61B8, "McuCANMsgService"},
                                                     //  {0x61BA, "SoCUdsService"},
                                                     //  {0x61BE, "SoCEgoMemService"},
                                                     //  {0x61C2, "McuADASRecordService"},
                                                     //  {0x61C4, "McuAEBRecordService"},
                                                     //  {0x61C6, "MCUmaintainService"},
                                                     //  {0x61C8, "TriggerIDService"},
                                                     //
                                                     {0x0001, "TEST"},
                                                     {0x0403, "ADASdataService"},
                                                     {0x0404, "APAdataService"},
                                                     {0, "NULL"}};

static const struct key_value SocDataService_id_values[] = {{0x8005, "/someip/socdataservice_1_0/TrajData"},
                                                            {0x8006, "/someip/socdataservice_1_0/PoseData"},
                                                            {0x8007, "/someip/socdataservice_1_0/SnsrFsnLaneDate"},
                                                            {0x8008, "/someip/socdataservice_1_0/SnsrFsnObj"},     
                                                            {0x800B, "/someip/socdataservice_1_0/ApaStateMachine"}, 
                                                            {0x800C, "/someip/socdataservice_1_0/AlgEgoToMCU"},
                                                            {0, "NULL"}};

static const struct key_value McuDataService_id_values[] = {{0x8005, "/someip/mcudataservice_1_0/MbdDebugData"},
                                                            {0x8006, "/someip/mcudataservice_1_0/AlgImuInsInfo"},
                                                            {0x8007, "/someip/mcudataservice_1_0/AlgGNSSPosInfo"},
                                                            {0x8008, "/someip/mcudataservice_1_0/AlgChassisInfo"},     
                                                            {0x8009, "/someip/mcudataservice_1_0/AlgPNCControl"}, 
                                                            {0x800A, "/someip/mcudataservice_1_0/AlgMcuToEgo"},
                                                            {0, "NULL"}};

static const struct key_value debug_id_values[] = {{0x0001, "debug_topic_1"},
                                                   {0x8101, "debug_topic_2"},
                                                   {0x8102, "debug_topic_3"},
                                                   {0x8103, "debug_topic_4"},
                                                   {0x8104, "debug_topic_5"},
                                                   {0x83e9, "debug_topic_6"},
                                                   {0x0000, "NULL"}};

#define RECORD_DEBUG false

#if RECORD_DEBUG == true
    #define debug_printf(format, ...) printf(format, ##__VA_ARGS__)
#else
    #define debug_printf(format, ...)
#endif


#ifdef __cplusplus
}
#endif

#endif /* NET_SOMEIP_H */
