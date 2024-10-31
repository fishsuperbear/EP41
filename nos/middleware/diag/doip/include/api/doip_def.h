/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip def
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_API_DOIP_DEF_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_API_DOIP_DEF_H_

#include <stdint.h>

namespace hozon {
namespace netaos {
namespace diag {

#define DOIP_TRUE 1
#define DOIP_FALSE 0


typedef enum DOIP_NETLINK_STATUS {
    DOIP_NETLINK_STATUS_UP       = 0x00,    /* doip serice is available */
    DOIP_NETLINK_STATUS_DOWN     = 0x01     /* doip serice not available */
} doip_netlink_status_t;

typedef enum DOIP_TA_TYPE {
    DOIP_TA_TYPE_PHYSICAL       = 0x00,    /* Physical addressing[One-on-one diagnosis] */
    DOIP_TA_TYPE_FUNCTIONAL     = 0x01     /* Functional addressing[Batch diagnosis] */
} doip_ta_type_t;

typedef enum DOIP_RESULT {
    DOIP_RESULT_OK                 = 0x00,    /* Success */
    DOIP_RESULT_HDR_ERROR          = 0x01,    /* Header data format error */
    DOIP_RESULT_INVALID_SA         = 0x02,    /* Source address is invalid */
    DOIP_RESULT_UNKNOWN_TA         = 0x03,    /* Target address is unknown */
    DOIP_RESULT_MESSAGE_TOO_LARGE  = 0x04,    /* Payload size too large */
    DOIP_RESULT_OUT_OF_MEMORY      = 0x05,    /* Insufficient buffer */
    DOIP_RESULT_TARGET_UNREACHABLE = 0x06,    /* Target address not be reached */
    DOIP_RESULT_NO_LINK            = 0x07,    /* No link */
    DOIP_RESULT_NO_SOCKET          = 0x08,    /* No socket */
    DOIP_RESULT_UNKNOWN_SA         = 0x09,    /* Source address is unknown */
    DOIP_RESULT_INITIAL_FAILED     = 0x0A,    /* Module init error */
    DOIP_RESULT_NOT_INITIALIZED    = 0x0B,    /* Call interface order error */
    DOIP_RESULT_ALREADY_INITED     = 0x0C,    /* Repeated initialization */
    DOIP_RESULT_PARAMETER_ERROR    = 0x0D,    /* The parameter is incorrect */
    DOIP_RESULT_REPEAT_REGIST      = 0x0E,    /* The SA is Repeat registration */
    DOIP_RESULT_CONFIG_ERROR       = 0x0F,    /* load config failed */
    DOIP_RESULT_TIMEOUT_A          = 0xA0,    /* Communication timeout */
    DOIP_RESULT_BUSY               = 0xA1,    /* Last diag message uncomplete */
    DOIP_RESULT_ERROR              = 0xFF     /* Common error */
} doip_result_t;

typedef enum DOIP_ENTITY_TYPE {
    DOIP_ENTITY_TYPE_GATEWAY         = 0x00,
    DOIP_ENTITY_TYPE_NODE            = 0x01,
    DOIP_ENTITY_TYPE_EDGE_GATEWAY    = 0x02,
    DOIP_ENTITY_TYPE_UNKNOWN         = 0xFF
} doip_entity_type_t;

typedef struct doip_request {
    uint16_t logical_source_address;    /* Unique identifier doip node. */
    uint16_t logical_target_address;    /* Unique identifier test euipment. */
    doip_ta_type_t ta_type;             /* Target address type. */
    char* data;                         /* UDS data[The TP is responsible for releasing the memory] */
    uint32_t data_length;               /* UDS data length */
} doip_request_t;

typedef struct doip_indication {
    uint16_t logical_source_address;    /* Unique identifier test euipment. */
    uint16_t logical_target_address;    /* Unique identifier doip node. */
    doip_ta_type_t ta_type;             /* Target address type. */
    char* data;                         /* RAW data[User needs to copy, DoIP is responsible for releasing memory] */
    uint32_t data_length;               /* RAW data length. */
    doip_result_t result;               /* Indication result */
} doip_indication_t;

typedef struct doip_confirm {
    uint16_t logical_source_address;    /* Unique identifier doip node. */
    uint16_t logical_target_address;    /* Unique identifier test euipment. */
    doip_ta_type_t ta_type;             /* Target address type. */
    doip_result_t result;               /* Confirm result */
} doip_confirm_t;

typedef struct doip_route {
    uint16_t logical_source_address;    /* Unique identifier test euipment. */
    uint16_t logical_target_address;    /* Unique identifier doip node. */
    doip_ta_type_t ta_type;             /* Target address type. */
    char* data;                         /* RAW data[User needs to copy, DoIP is responsible for releasing memory] */
    uint32_t data_length;               /* RAW data length. */
    doip_result_t result;               /* Indication result */
} doip_route_t;


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_API_DOIP_DEF_H_
