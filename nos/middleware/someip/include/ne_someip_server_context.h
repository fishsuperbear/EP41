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
#ifndef MANAGER_SERVER_NE_SOMEIP_SERVER_H
#define MANAGER_SERVER_NE_SOMEIP_SERVER_H

#ifdef  __cplusplus
extern "C" {
#endif

#include "ne_someip_handler.h"
#include "ne_someip_config_define.h"
#include "ne_someip_looper.h"

typedef struct ne_someip_server_context ne_someip_server_context_t;

/**
 *@brief reference the server object, the object reference count will increase by 1.
 *
 *@param [in] server, the server object.
 *
 *@return, the server object.
 *
 *@attention Synchronous I/F.
 */
ne_someip_server_context_t* ne_someip_server_ref(ne_someip_server_context_t* server);

/**
 *@brief unreference the server object, the object reference count will decrease by 1.
 *
 *@param [in] server, the server object.
 *
 *@attention Synchronous I/F.
 */
void ne_someip_server_unref(ne_someip_server_context_t* server);

/**
 *@brief create the server object. the object reference count will be set to 1.
 *
 *@param [in] is_shared_thread, if true, this server will share one thread with other client, if false, this server uses exclusive thread.
 *@param [in] priority, the server thread priority.
 *@param [in] schedule, the thread schedule.
 *
 *@return, the server object.
 *
 *@attention Synchronous I/F.
 */
ne_someip_server_context_t* ne_someip_server_create(bool is_shared_thread, int priority, int schedule);

/**
 *@brief create instance.
 *
 *@param [in] server, the server object.
 *@param [in] instance_config, the instance config.
 *
 *@return, the instance object.
 *
 *@attention Synchronous I/F.
 */
ne_someip_provided_instance_t* ne_someip_server_create_instance(ne_someip_server_context_t* server,
	const ne_someip_provided_service_instance_config_t* instance_config);

/**
 *@brief destroy instance, the instance will stop all behaviour and be null.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *@return, the instance object.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_destroy_instance(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief offer service.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *maybe any error or success occured during the request processed, someip client can received callback by call
 *ne_someip_server_reg_service_status_handler function which is defined in include/extend/someip/ne_someip_server_context.h.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Asynchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_offer(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief stop offer service
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *maybe any error or success occured during the request processed, someip client can received callback by call
 *ne_someip_server_reg_service_status_handler function which is defined in include/extend/someip/ne_someip_server_context.h.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Asynchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_stop_offer(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief send event.
 *
 *@param [in] server, the server object.
 *@param [in] seq_id, the sequence id.
 *@param [in] instance, the instance object.
 *@param [in] header, the someip header message.
 *@param [in] payload, the someip payload message.
 *
 *maybe any error or success occured during the request processed, someip client can received callback by call
 *ne_someip_server_reg_event_status_handler function which is defined in include/extend/someip/ne_someip_server_context.h.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Asynchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_send_event(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, const void* seq_id,
	ne_someip_header_t* header, ne_someip_payload_t* payload);

/**
 *@brief send response.
 *
 *@param [in] server, the server object.
 *@param [in] seq_id, the sequence id.
 *@param [in] instance, the instance object.
 *@param [in] header, the someip header message.
 *@param [in] payload, the someip payload message.
 *@param [in] remote_addr, the remote address.
 *
 *maybe any error or success occured during the request processed, someip client can received callback by call
 *ne_someip_server_reg_response_status_handler function which is defined in include/extend/someip/ne_someip_server_context.h.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Asynchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_send_response(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, const void* seq_id,
	ne_someip_header_t* header, ne_someip_payload_t* payload,
	const ne_someip_remote_client_info_t* remote_addr);

/**
 *@brief register request handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *@param [in] handler, the handler that receives the request from remote client.
 *@param [in] user_data, the user data of client.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_reg_req_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, ne_someip_recv_request_handler handler, const void* user_data);

/**
 *@brief unregister request handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_unreg_req_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief register subscribe handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *@param [in] handler, the handler that receives the subscribes from remote client.
 *@param [in] user_data, the user data of client.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_reg_subscribe_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, ne_someip_recv_subscribe_handler handler, const void* user_data);

/**
 *@brief unregister subscribe handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_unreg_subscribe_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief register service status handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *@param [in] handler, the handler that receives the service status and error code.
 *@param [in] user_data, the user data of client.
 *
 *############################################
 *if ne_someip_offer_status_handler call back error code is ne_someip_error_code_network_down, the client does not need to retry offer
 *############################################
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_reg_service_status_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, ne_someip_offer_status_handler handler, const void* user_data);

/**
 *@brief unregister service status handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_unreg_service_status_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief register event handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *@param [in] handler, the handler that receives the send result of event.
 *@param [in] user_data, the user data of client.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_reg_event_status_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, ne_someip_send_event_status_handler handler, const void* user_data);

/**
 *@brief unregister event handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_unreg_event_status_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief register response handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *@param [in] handler, the handler that receives the send result of response.
 *@param [in] user_data, the user data of client.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_reg_resp_status_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, ne_someip_send_resp_status_handler handler, const void* user_data);

/**
 *@brief unregister response handler.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_unreg_resp_status_handler(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief get service status.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_offer_status_t ne_someip_server_get_service_status(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance);

/**
 *@brief get service status.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *@param [in] eventgroup_id, the eventgroup id.
 *@param [in] remote_addr, the remote address.
 *@param [in] priority, the priority of eventgroup for remote.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_set_eventgroup_permission(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, ne_someip_eventgroup_id_t eventgroup_id,
	const ne_someip_remote_client_info_t* remote_addr, ne_someip_permission_t priority);

/**
 *@brief get service status.
 *
 *@param [in] server, the server object.
 *@param [in] instance, the instance object.
 *@param [in] method_id, the method id.
 *@param [in] remote_addr, the remote address.
 *@param [in] priority, the priority of method for remote.
 *
 *@return, ne_someip_error_code_ok or ne_someip_error_code_failed.
 *
 *@attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_server_set_method_permission(ne_someip_server_context_t* server,
	ne_someip_provided_instance_t* instance, ne_someip_method_id_t method_id,
	const ne_someip_remote_client_info_t* remote_addr, ne_someip_permission_t priority);

/**
 * @brief get the work looper.
 * 
 * @param [in] server, the server object.
 * 
 * @return work looper pointer.
 * 
 * @attention Synchronous I/F.
 */
ne_someip_looper_t* ne_someip_server_get_work_looper(ne_someip_server_context_t* server);

/**
 *@brief create response header.
 *
 *@param [in] req_header, the someip header message.
 *@param [out] resp_header, the response header.
 *@param [in] message_type, the message type.
 *@param [in] return_code, the return code.
 *
 *@return.
 *
 *@attention Synchronous I/F.
 */
void ne_someip_server_create_resp_header(const ne_someip_header_t* req_header,
	ne_someip_header_t* resp_header, ne_someip_message_type_t message_type, ne_someip_return_code_t return_code);

/**
 *@brief create event header.
 *
 *@param [in] instance, the instance object.
 *@param [in] event_id, the event id.
 *@param [out] header, the event header.
 *
 *@return.
 *
 *@attention Synchronous I/F.
 */
void ne_someip_server_create_notify_header(ne_someip_provided_instance_t* instance,
	ne_someip_event_id_t event_id, ne_someip_header_t* header);

/**
 *@brief create event header with session id.
 *
 *@param [in] instance, the instance object.
 *@param [in] event_id, the event id.
 *@param [in] session_id, the session id from user.
 *@param [out] header, the event header.
 *
 *@return.
 *
 *@attention Synchronous I/F.
 */
void ne_someip_server_create_notify_header_with_session(ne_someip_provided_instance_t* instance,
	ne_someip_event_id_t event_id, ne_someip_session_id_t session_id, ne_someip_header_t* header);

/**
 *@brief get the service instance info from instance object.
 *
 *@param [in] instance, the instance object.
 *@param [out] ins_spec, the service id, instance id and major version.
 *
 *@return, true or false.
 *
 *@attention Synchronous I/F.
 */
bool ne_someip_server_get_instance_info(const ne_someip_provided_instance_t* instance,
	ne_someip_service_instance_spec_t* ins_spec);

#ifdef __cplusplus
}
#endif
#endif
/* EOF */