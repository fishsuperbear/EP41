/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/

#ifndef INCLUDE_NE_SOMEIP_DEFINE_H
#define INCLUDE_NE_SOMEIP_DEFINE_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "ne_someip_object.h"

#define NESOMEIP_FILE_NAME_LENGTH 108
#define NESOMEIP_IF_NAME_LENGTH 16
#define NESOMEIP_IPv6_ADDR_LENGTH 17
#define NESOMEIP_IP_ADDR_LENGTH 17
#define NESOMEIP_DEFAULT_MESSAGE_LENGTH 8

// Define types for SOME/IP message header
typedef uint32_t ne_someip_message_id_t;

typedef uint16_t ne_someip_service_id_t;
typedef uint16_t ne_someip_method_id_t;
typedef uint16_t ne_someip_event_id_t;

typedef uint32_t ne_someip_request_id_t;
typedef uint16_t ne_someip_client_id_t;
typedef uint16_t ne_someip_session_id_t;
typedef uint8_t ne_someip_message_type_t;

typedef uint8_t ne_someip_protocol_version_t;
typedef uint8_t ne_someip_interface_version_t;

typedef uint16_t ne_someip_instance_id_t;
typedef uint32_t ne_someip_ttl_t;

typedef uint8_t ne_someip_major_version_t;
typedef uint32_t ne_someip_minor_version_t;

typedef uint16_t ne_someip_eventgroup_id_t;
typedef uint8_t ne_someip_return_code_t;
typedef uint32_t ne_someip_message_length_t;
typedef uint8_t ne_someip_byte_t;

typedef uint32_t ne_someip_sequence_id_t;

typedef struct ne_someip_client_context ne_someip_client_context_t;
typedef struct ne_someip_required_service_instance ne_someip_required_service_instance_t;
typedef struct ne_someip_provided_service_instance ne_someip_provided_instance_t;

// Define the error code for function call
typedef enum ne_someip_error_code
{
    ne_someip_error_code_ok = 0x00,  // Synchronous success.
    ne_someip_error_code_failed = 0x01,  // Synchronous failed.
    ne_someip_error_code_create_udp_error = 0x02,  // create udp endpoint failed.
    ne_someip_error_code_create_tcp_error = 0x03,  // create tcp endpoint failed.
    ne_someip_error_code_tcp_connect_error = 0x4,  // tcp server disconnect by tcp client.
    ne_someip_error_code_no_offer_before = 0x05,  // not offer service before. client does not need to retry.
    ne_someip_error_code_network_down = 0x06,  // network card down. client does not need to retry.
    ne_someip_error_code_system_error = 0x07, //system error.
    ne_someip_error_code_ttl_timeout = 0x08, // ttl timeout.
    ne_someip_error_code_no_e2e_protect = 0x09, // no e2e protect in config.
    ne_someip_error_code_payload_too_long = 0x0A, // someip data length longer than tcp/unix buffer size
    ne_someip_error_code_unix_disconnect = 0x0B,
    ne_someip_error_code_sync_call_timeout = 0x0C, // sync call timeout
    ne_someip_error_code_unknown  = 0xFF,
} ne_someip_error_code_t;

/* message type enum define */
typedef enum ne_someip_message_type_enum
{
    ne_someip_message_type_request = 0x00,
    ne_someip_message_type_request_no_return = 0x01,
    ne_someip_message_type_notification = 0x02,
    ne_someip_message_type_response = 0x80,
    ne_someip_message_type_error = 0x81,
    ne_someip_message_type_unknown = 0xFF,
} ne_someip_message_type_enum_t;

/* return code for someip header */
enum ne_someip_return_code
{
    ne_someip_return_code_ok = 0x00, //no error
    ne_someip_return_code_not_ok = 0x01, //an unspecified error
    ne_someip_return_code_unkown_service = 0x02, //unkown service id
    ne_someip_return_code_unkown_method = 0x03, //unkown method id
    ne_someip_return_code_not_ready = 0x04, //application not running
    ne_someip_return_code_not_reachable = 0x05, //service is not reachable(internal error)
    ne_someip_return_code_timeout = 0x06, //timeout(internal error)
    ne_someip_return_code_wrong_protocol_version = 0x07, //not supported protocol version
    ne_someip_return_code_wrong_interface_version = 0x08, //not supported interface version
    ne_someip_return_code_malformed_msg = 0x09, //deserialize error
    ne_someip_return_code_wrong_msg_type = 0x0A, //wrong message type
    ne_someip_return_code_generic_error_reserved_min = 0x0B, //reserved for generic SOME/IP errors
    ne_someip_return_code_generic_error_reserved_max = 0x1F, //reserved for generic SOME/IP errors
    ne_someip_return_code_specific_error_reserved_min = 0x20, //reserved for specific errors of services and methods
    ne_someip_return_code_specific_error_reserved_max = 0x5E, //reserved for specific errors of services and methods
};

// define protocol that used by someip in transport layer
typedef enum ne_someip_l4_protocol
{
    ne_someip_protocol_unknown    = 0x0,    /* initialization value */
    ne_someip_protocol_udp        = 0x11,    /* UDP protocol */
    ne_someip_protocol_tcp        = 0x06,    /* TCP protocol */
    ne_someip_protocol_tcp_udp    = 0x17,    /* TCP and UDP protocol */
} ne_someip_l4_protocol_t;

// define address type
typedef enum ne_someip_address_type
{
    ne_someip_address_type_unknown = 0x0,
    ne_someip_address_type_ipv4 = 0x04,
    ne_someip_address_type_ipv6 = 0x06,
}ne_someip_address_type_t;

typedef enum ne_someip_service_status
{
    ne_someip_service_status_available = 0x0,
    ne_someip_service_status_unavailable = 0x1,
}ne_someip_service_status_t;

typedef enum ne_someip_find_status
{
    ne_someip_find_status_stopped = 0x00,
    ne_someip_find_status_pending = 0x01,
    ne_someip_find_status_running = 0x02,
}ne_someip_find_status_t;

typedef enum ne_someip_subscribe_status
{
    ne_someip_subscribe_status_unsubscribed = 0x00,
    ne_someip_subscribe_status_pending = 0x01,
    ne_someip_subscribe_status_subscribed = 0x02,
    ne_someip_subscribe_status_failed = 0x03,
}ne_someip_subscribe_status_t;

typedef enum ne_someip_offer_status
{
    ne_someip_offer_status_stopped = 0x00,
    ne_someip_offer_status_pending = 0x01,
    ne_someip_offer_status_running = 0x02,
}ne_someip_offer_status_t;

typedef enum ne_someip_permission
{
    ne_someip_permission_unkown = 0x0,
    ne_someip_permission_reject = 0x01,
    ne_someip_permission_allow = 0x02,
} ne_someip_permission_t;

typedef enum ne_someip_log_type
{
    NE_SOMEIP_LOG_TYPE_DEBUG   = 1,
    NE_SOMEIP_LOG_TYPE_INFO,
    NE_SOMEIP_LOG_TYPE_WARNING,
    NE_SOMEIP_LOG_TYPE_ERROR
} ne_someip_log_type_t;

typedef enum ne_someip_tls_version
{
    ne_someip_tls_version_1_2 = 0x01,
    ne_someip_tls_version_1_3 = 0x02,
} ne_someip_tls_version_t;

typedef struct ne_someip_payload_slice
{
    void* free_pointer;
    uint8_t* data;
    ne_someip_message_length_t length;
}ne_someip_payload_slice_t;

// define someip payload info struct
typedef struct ne_someip_payload
{
    ne_someip_payload_slice_t** buffer_list;
    uint32_t num;
    NEOBJECT_MEMBER
} ne_someip_payload_t;

// define someip message header info struct
typedef struct ne_someip_header
{
    ne_someip_service_id_t service_id;                 /* service id in someip message */
    ne_someip_method_id_t method_id;                   /* instance id in someip message */
    ne_someip_message_length_t message_length;         /* message length in someip message */
    ne_someip_client_id_t client_id;                   /* client id in someip message */
    ne_someip_session_id_t session_id;                 /* session id in someip message */
    ne_someip_protocol_version_t protocol_version;     /* protocol version in someip message */
    ne_someip_interface_version_t interface_version;   /* interface version in someip message */
    ne_someip_message_type_t message_type;             /* message type in someip message */
    ne_someip_return_code_t return_code;
} ne_someip_header_t;


// define remote client info
typedef struct ne_someip_remote_client_info
{
    ne_someip_address_type_t type;
    uint32_t ipv4;
    char ipv6[NESOMEIP_IPv6_ADDR_LENGTH];
    uint16_t port;
    ne_someip_l4_protocol_t protocol;
} ne_someip_remote_client_info_t;

// The struct that was used to indicate service instance (client and server)
typedef struct ne_someip_service_instance_spec
{
    ne_someip_service_id_t service_id;
    ne_someip_instance_id_t instance_id;
    ne_someip_major_version_t major_version;
}ne_someip_service_instance_spec_t;

// used to find service in someip stack
typedef struct ne_someip_find_offer_service_spec
{
    ne_someip_service_instance_spec_t ins_spec;
    ne_someip_minor_version_t minor_version;
} ne_someip_find_offer_service_spec_t;

typedef struct ne_someip_eventgroup_id_list
{
    ne_someip_eventgroup_id_t* eventgroup_id_list;
    uint32_t num;
} ne_someip_eventgroup_id_list_t;

typedef struct ne_someip_find_local_offer_services
{
    ne_someip_find_offer_service_spec_t* local_services;
    uint32_t local_services_num;
} ne_someip_find_local_offer_services_t;

/**
 *@brief create payload, the payload reference count will be 1, the payload slice will be free by payload.
 *
 *@return the payload.
 */
ne_someip_payload_t* ne_someip_payload_create();

/**
 *@brief reference payload.
 *
 *@param [in] payload, the payload.
 *
 *@return the payload.
 */
ne_someip_payload_t* ne_someip_payload_ref(ne_someip_payload_t* payload);

/**
 *@brief unreference payload.
 *
 *@param [in] payload, the payload.
 *
 */
void ne_someip_payload_unref(ne_someip_payload_t* payload);

/**
* @brief [SomeIp Stack server use] printf log callback function.
*
* @param [in] text : printf data.
*
* @attention The pointer is released by the someip module.
*/
typedef void (*ne_someip_logfunc_callback) (ne_someip_log_type_t type, const char *logTag, const char *text);

void ne_someip_set_logfunc_callback(ne_someip_logfunc_callback logfunc);

#ifdef __cplusplus
}
#endif
#endif // INCLUDE_NE_SOMEIP_DEFINE_H
/* EOF */
