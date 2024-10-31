/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip internal def
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_DATA_TYPE_DOIP_DEF_INTERNAL_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_DATA_TYPE_DOIP_DEF_INTERNAL_H_

#include <stdint.h>
#include <netinet/in.h>
#include <memory>

#include "diag/doip/include/api/doip_def.h"
#include "diag/doip/include/base/doip_data_queue.h"

namespace hozon {
namespace netaos {
namespace diag {


#define DOIP_VIN_SIZE            17   /* Vehicle identification number size */
#define DOIP_EID_SIZE            6    /* Entify iedntification size */
#define DOIP_GID_SIZE            6    /* Group identification size */
#define DOIP_IFNAME_SIZE         16   /* netowrk interface name size identification size */

#define DOIP_TCP_DATA_PORT       13400
#define DOIP_UDP_DISCOVERY_PORT  13400
#define UDP_TEST_EQUIPMENT_REQUEST_MIN_PORT 49152
#define UDP_TEST_EQUIPMENT_REQUEST_MAX_PORT 65535
#define DOIP_UDP_BROADCAST_IP    "255.255.255.255"
#define DOIP_UDP6_BROADCAST_IP   "FF02::1"

#define DOIP_A_DOIP_ANNOUNCE_WAIT_MAX   500U      /* 500 ms */
#define DOIP_A_DOIP_ANNOUNCE_INTERVAL   500U      /* 500 ms */
#define DOIP_T_TCP_INITIAL_INACTIVITY   2000U     /* 2 s    */
#define DOIP_T_TCP_GENE_INACTIVITY      300000U   /* 5 min  */
#define DOIP_T_TCP_ALIVE_CHECK          500U      /* 500 ms */

#define DOIP_DEFAULT_VERSION 0xFFU
#define DOIP_CURRENT_VERSION 0x02U

#define DOIP_NO_FURTHER_ACTION_REQUIRED  0x00U
#define DOIP_FURTHER_ACTION_REQUIRED     0x10U

#define DOIP_ST_PAYLOADTYPE_VEHICLE_INDENTIFY         0x0001U
#define DOIP_ST_PAYLOADTYPE_VEHICLE_INDENTIFY_EID     0x0002U
#define DOIP_ST_PAYLOADTYPE_VEHICLE_INDENTIFY_VIN     0x0003U
#define DOIP_ST_PAYLOADTYPE_ROUTING_ACTIVE_REQ        0x0005U
#define DOIP_ST_PAYLOADTYPE_ALIVE_CHECK_RES           0x0008U
#define DOIP_ST_PAYLOADTYPE_ENTITY_STATUS_REQ         0x4001U
#define DOIP_ST_PAYLOADTYPE_POWER_MODE_REQ            0x4003U
#define DOIP_ST_PAYLOADTYPE_DIAGNOSTIC_REQ            0x8001U
#define DOIP_ST_PAYLOADTYPE_HEADER_NEGTIVE_ACK        0x0000U
#define DOIP_ST_PAYLOADTYPE_ANNOUNCE_OR_IDENTIFYRES   0x0004U
#define DOIP_ST_PAYLOADTYPE_ROUTING_ACTIVE_RES        0x0006U
#define DOIP_ST_PAYLOADTYPE_ALIVE_CHECK_REQ           0x0007U
#define DOIP_ST_PAYLOADTYPE_ENTITY_STATUS_RES         0x4002U
#define DOIP_ST_PAYLOADTYPE_POWER_MODE_RES            0x4004U
#define DOIP_ST_PAYLOADTYPE_DIAG_POSITIVE_ACK         0x8002U
#define DOIP_ST_PAYLOADTYPE_DIAG_NEGATIVE_ACK         0x8003U

#define DOIP_LOGICAL_ADDRESS_LENGTH 2
#define DOIP_CONF_RESULT_LENGTH     1
#define DOIP_TA_TYPE_LENGTH         1
#define DOIP_BLANK_2_LENGTH         2
#define DOIP_FURTHER_ACTION_LENGTH  1
#define DOIP_VIN_GID_SYNC_LENGTH    1
#define DOIP_RA_RES_CODE_LENGTH     1
#define DOIP_NODE_TYPE_LENGTH       1
#define DOIP_MCTS_LENGTH            1
#define DOIP_NCTS_LENGTH            1
#define DOIP_MDS_LENGTH             4
#define DOIP_POWERMODE_LENGTH       1
#define DOIP_ACK_CODE_LENGTH        1
#define DOIP_NACK_CODE_LENGTH       1
#define DOIP_ACTIVATION_TYPE_LENGTH 1
#define DOIP_RESERVED_LENGTH        4
#define DOIP_OEM_SPECIFIC_LENGTH    4
#define DOIP_PROTOCOL_VERSION_LENGTH 1
#define DOIP_INVERSE_PROTOCOL_VERSION_LENGTH 1
#define DOIP_PAYLOAD_TYPE_LENGTH    2
#define DOIP_PAYLOAD_LENGTH_LENGTH  4
#define DOIP_HEADER_COMMON_LENGTH   8
#define DOIP_ANNOUNCE_OR_IDENTITYRES_MAND_LENGTH  DOIP_EID_SIZE + \
                                                  DOIP_GID_SIZE + \
                                                  DOIP_VIN_SIZE + \
                                                  DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_FURTHER_ACTION_LENGTH
#define DOIP_ANNOUNCE_OR_IDENTITYRES_ALL_LENGTH   DOIP_EID_SIZE + \
                                                  DOIP_GID_SIZE + \
                                                  DOIP_VIN_SIZE + \
                                                  DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_FURTHER_ACTION_LENGTH + \
                                                  DOIP_VIN_GID_SYNC_LENGTH
#define DOIP_ROUTING_ACTIVATION_RES_MAND_LENGTH   DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_RA_RES_CODE_LENGTH +\
                                                  DOIP_RESERVED_LENGTH
#define DOIP_ROUTING_ACTIVATION_RES_ALL_LENGTH    DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_RA_RES_CODE_LENGTH +\
                                                  DOIP_RESERVED_LENGTH + \
                                                  DOIP_OEM_SPECIFIC_LENGTH
#define DOIP_ENTITY_STATUS_RES_MAND_LENGTH        DOIP_NODE_TYPE_LENGTH + \
                                                  DOIP_MCTS_LENGTH + \
                                                  DOIP_NCTS_LENGTH
#define DOIP_ENTITY_STATUS_RES_ALL_LENGTH         DOIP_NODE_TYPE_LENGTH + \
                                                  DOIP_MCTS_LENGTH + \
                                                  DOIP_NCTS_LENGTH + \
                                                  DOIP_MDS_LENGTH
#define DOIP_HEADER_NEGATIVE_ACK_LENGTH           DOIP_NACK_CODE_LENGTH
#define DOIP_POWERMODE_INFO_RES_LENGTH            DOIP_POWERMODE_LENGTH
#define DOIP_DIAG_POSITIVE_ACK_LENGTH             DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_ACK_CODE_LENGTH
#define DOIP_DIAG_NEGATIVE_ACK_LENGTH             DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_NACK_CODE_LENGTH
#define DOIP_ROUTING_ACTIVATION_REQ_MAND_LENGTH   DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_ACTIVATION_TYPE_LENGTH + \
                                                  DOIP_RESERVED_LENGTH
#define DOIP_ROUTING_ACTIVATION_REQ_ALL_LENGTH    DOIP_LOGICAL_ADDRESS_LENGTH + \
                                                  DOIP_ACTIVATION_TYPE_LENGTH + \
                                                  DOIP_RESERVED_LENGTH + \
                                                  DOIP_OEM_SPECIFIC_LENGTH


typedef enum DOIP_PAYLOAD_TYPE_INFO {
    DOIP_PAYLOAD_TYPE_HEADER_NEGATIVE_ACK          = 0x00,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_VEHICLE_ANNOUNCEMENT         = 0x01,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_VEHICLE_IDENTIFY_RESPONSE    = 0x02,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_ROUTING_ACTIVE_RESPONSE      = 0x03,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_ALIVE_CHECK_REQUEST          = 0x04,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_ENTITY_STATUS_RESPONSE       = 0x05,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_POWER_MODE_INFO_RESPONSE     = 0x06,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_DIAGNOSTIC_POSITIVE_ACK      = 0x07,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_DIAGNOSTIC_NEGATIVE_ACK      = 0x08,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_DIAGNOSTIC_FROM_ENTITY       = 0x09,   // ---->for doip node
    DOIP_PAYLOAD_TYPE_VEHICLE_IDENTIFY_REQUEST     = 0x10,   // ---->for internal equip
    DOIP_PAYLOAD_TYPE_VEHICLE_IDENTIFY_REQUEST_EID = 0x11,   // ---->for internal equip
    DOIP_PAYLOAD_TYPE_VEHICLE_IDENTIFY_REQUEST_VIN = 0x12,   // ---->for internal equip
    DOIP_PAYLOAD_TYPE_ROUTING_ACTIVE_REQUEST       = 0x13,   // ---->for internal equip
    DOIP_PAYLOAD_TYPE_ALIVE_CHECK_RESPONSE         = 0x14,   // ---->for internal equip
    DOIP_PAYLOAD_TYPE_ENTITY_STATUS_REQUEST        = 0x15,   // ---->for internal equip
    DOIP_PAYLOAD_TYPE_POWER_MODE_INFO_REQUEST      = 0x16,   // ---->for internal equip
    DOIP_PAYLOAD_TYPE_DIAGNOSTIC_FROM_EQUIP        = 0x17,   // ---->for internal equip
} doip_payload_type_info_t;

typedef enum DOIP_HEADER_NACK_CODE {
    DOIP_HEADER_NACK_INCORRECT_PATTERN_FORMAT = 0x00, /* mandatory */
    DOIP_HEADER_NACK_UNKNOWN_PAYLOAD_TYPE     = 0x01, /* mandatory */
    DOIP_HEADER_NACK_MESSAGE_TOO_LARGE        = 0x02, /* mandatory */
    DOIP_HEADER_NACK_OUT_OF_MEMORY            = 0x03, /* mandatory */
    DOIP_HEADER_NACK_INVALID_PAYLOAD_LENGTH   = 0x04, /* mandatory */
} doip_header_nack_code_t;

typedef enum DOIP_DIAGNOSTIC_NACK_CODE {
    DOIP_DIAGNOSTIC_NACK_INVALID_SA                  = 0x02, /* mandatory */
    DOIP_DIAGNOSTIC_NACK_UNKNOWN_TA                  = 0x03, /* mandatory */
    DOIP_DIAGNOSTIC_NACK_DIAG_MSG_TOO_LARGE          = 0x04, /* mandatory */
    DOIP_DIAGNOSTIC_NACK_OUT_OF_MEMORY               = 0x05, /* mandatory */
    DOIP_DIAGNOSTIC_NACK_TARGET_UNREACHABLE          = 0x06, /* optional */
    DOIP_DIAGNOSTIC_NACK_UNKNOWN_NETWORK             = 0x07, /* optional */
    DOIP_DIAGNOSTIC_NACK_TRANS_PROTO_ERROR           = 0x08, /* optional */
    DOIP_DIAGNOSTIC_NACK_ACK_TIMEOUT                 = 0xA0, /* customized */
    DOIP_DIAGNOSTIC_NACK_LAST_DIAGNOSTIC_UNCOMPLETE  = 0xA1, /* customized */
} doip_diagnostic_nack_code_t;

typedef enum DOIP_ROUTING_ACTIVE_RES_CODE {
    DOIP_RA_RES_UNKNOWN_SOURCE_ADDRESS              = 0x00, /* mandatory */
    DOIP_RA_RES_ALL_SOCKET_REGISTED_AND_ACTIVE      = 0x01, /* mandatory */
    DOIP_RA_RES_SA_DIFFERENT_ACTIVATED_SOCKET       = 0x02, /* mandatory */
    DOIP_RA_RES_SA_REGISTED_DIFFERENT_SOCKET        = 0x03, /* mandatory */
    DOIP_RA_RES_MISSING_AUTHENTICATION              = 0x04, /* optional */
    DOIP_RA_RES_REJECT_CONFIRMATION                 = 0x05, /* optional */
    DOIP_RA_RES_UNSUPPORTED_RA_TYPE                 = 0x06, /* mandatory */
    DOIP_RA_RES_ROUTING_SUCCESSFULLY_ACTIVATED      = 0x10, /* mandatory */
    DOIP_RA_RES_CONFIRMATION_REQUIRED               = 0x11, /* optional */
} doip_routing_activation_res_code_t;

typedef enum DOIP_ROUTING_ACTIVATION_STEP {
    DOIP_ROUTING_ACTIVATION_STEP_INITIALIZED = 0x00,
    DOIP_ROUTING_ACTIVATION_STEP_PENDING     = 0x01,
    DOIP_ROUTING_ACTIVATION_STEP_REGISTED    = 0x02,
} doip_routing_activation_step_t;

typedef enum DOIP_ALIVE_CHECK_STATUS {
    DOIP_ALIVE_CHECK_STATUS_NONE    = 0x00,
    DOIP_ALIVE_CHECK_STATUS_ALL     = 0x01,
    DOIP_ALIVE_CHECK_STATUS_SINGLE  = 0x02,
} doip_alive_check_status_t;

typedef enum DOIP_CONNECT_STATE {
    DOIP_CONNECT_STATE_LISTEN                      = 0x00,
    DOIP_CONNECT_STATE_INITIALIZED                 = 0x01,
    DOIP_CONNECT_STATE_REGISTERED_PENDING_FOR_AUTH = 0x02,
    DOIP_CONNECT_STATE_REGISTERED_PENDING_FOR_CONF = 0x03,
    DOIP_CONNECT_STATE_REGISTERED_ROUTING_ACTIVE   = 0x04,
    DOIP_CONNECT_STATE_FINALIZE                    = 0x05
} doip_connect_state_t;

typedef enum DOIP_CLIENT_TYPE {
    DOIP_CLIENT_TYPE_IPC        = 0x01,    /* ipc client type */
    DOIP_CLIENT_TYPE_UDP_SERVER = 0x02,    /* udpserver client type */
    DOIP_CLIENT_TYPE_UDP_CLIENT = 0x03,    /* udpclient client type */
    DOIP_CLIENT_TYPE_TCP_SERVER = 0x04,    /* tcpserver client type */
    DOIP_CLIENT_TYPE_TCP_CLIENT = 0x05,    /* tcpclient client type  old name: test client */
    DOIP_CLIENT_TYPE_FREE       = 0x0F     /* free client type */
} doip_clinet_type;

typedef enum DOIP_SOCKET_TYPE {
    DOIP_SOCKET_TYPE_UNKNOWN    = 0x00,
    DOIP_SOCKET_TYPE_IPC        = 0x01,
    DOIP_SOCKET_TYPE_UDP_SERVER = 0x02,
    DOIP_SOCKET_TYPE_UDP_CLIENT = 0x03,
    DOIP_SOCKET_TYPE_TCP_SERVER = 0x04,
    DOIP_SOCKET_TYPE_TCP_CLIENT = 0x05,
    DOIP_SOCKET_TYPE_SELECT     = 0x06,
} doip_socket_type;

typedef enum DOIP_NODE_TYPE {
    DOIP_NT_GATEWAY = 0x00,    /* Node type is gateway */
    DOIP_NT_NODE    = 0x01,    /* Node type is node    */
    DOIP_NT_UNKOWN  = 0xFF     /* Node type is unkown  */
} doip_node_type_t;

typedef struct doip_node_tcp_table {
    char ip[INET6_ADDRSTRLEN];
    uint16_t port;
    int32_t fd;
    uint16_t equip_logical_address;
    int32_t tcp_initial_inactivity_timer_fd;
    int32_t tcp_generral_inactivity_timer_fd;
    int32_t tcp_alive_check_timer_fd;
    uint32_t OEM_specific;
    uint8_t activation_type;
    doip_connect_state_t connection_state;
    char header_cache_buffer[DOIP_HEADER_COMMON_LENGTH];
    char* payload_cache_buffer;
    uint32_t header_current_pos;
    uint32_t payload_current_pos;
    uint8_t payload_ignore_flag;
} doip_node_tcp_table_t;

typedef struct doip_node_udp_table {
    char ip[INET6_ADDRSTRLEN];
    uint16_t port;
    int32_t fd;
    uint16_t logical_address;
    int32_t vehicle_identify_wait_timer_fd;
    char header_cache_buffer[DOIP_HEADER_COMMON_LENGTH];
    char* payload_cache_buffer;
} doip_node_udp_table_t;

typedef struct doip_equip_tcp_table {
    int32_t fd;
    char ip[INET6_ADDRSTRLEN];
    uint16_t port;
    uint16_t entity_logical_address;
    uint16_t equip_logical_address;
    int32_t ack_timer_fd;
    bool is_ra_res_confirming;
    bool is_diag_ack_confirming;
    doip_routing_activation_step_t routing_activation_step;
    char header_cache_buffer[DOIP_HEADER_COMMON_LENGTH];
    char* payload_cache_buffer;
    uint32_t header_current_pos;
    uint32_t payload_current_pos;
    uint8_t payload_ignore_flag;
    uint8_t role;
    DoipDataQueue* doip_data_queue;
} doip_equip_tcp_table_t;

typedef struct doip_equip_udp_table {
    int32_t fd;
    char ip[INET6_ADDRSTRLEN];
    uint16_t port;
    uint16_t logical_address;
    char eid[DOIP_EID_SIZE];
    char header_cache_buffer[DOIP_HEADER_COMMON_LENGTH];
    char* payload_cache_buffer;
} doip_equip_udp_table_t;

typedef struct doip_alive_check_originator {
    doip_alive_check_status_t alive_check_status;
    uint8_t alive_check_ncts;
    uint8_t is_res_doing;
    doip_node_tcp_table_t* originator;
} doip_alive_check_originator_t;

typedef struct doip_link_data {
    int32_t fd;
    char ip[INET6_ADDRSTRLEN];
    uint16_t port;
    uint8_t comm_type;
    char *data;
    uint32_t data_size;
} doip_link_data_t;


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_DATA_TYPE_DOIP_DEF_INTERNAL_H_
