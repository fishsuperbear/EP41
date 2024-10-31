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
#ifndef SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_COMMON_FUNC_H
#define SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_COMMON_FUNC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ne_someip_endpoint_define.h"

/**
 * @brief The register instance pointer interface in endpoint used by required-service-instance side. (called and run in work thread)
 *        Only used for unix endpoint in proxy
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] client_id : the client id of the called service-instance
 * @param [in] service_instance_key : service-id, instance-id and major-version of the called service-instance
 * @param [in] service_instance : the void service-instance pointer. can convert to specific service-instance
 * @param [in] type : the type of service_instance
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_register_instance_client(void* endpoint, ne_someip_client_id_t client_id,
    ne_someip_service_instance_spec_t* service_instance_key, const void* service_instance, ne_someip_endpoint_instance_type_t type);

/**
 * @brief The unregister instance pointer interface in endpoint used by required-service-instance side. (called and run in work thread)
 *        Only used for unix endpoint in proxy
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] client_id : the client id of the called service-instance
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unregister_instance_client(void* endpoint, ne_someip_client_id_t client_id,
    ne_someip_service_instance_spec_t* service_instance_key, const void* service_instance, ne_someip_endpoint_instance_type_t type);

/**
 * @brief The register instance pointer interface in endpoint used by provided-service-instance side. (called and run in work thread)
 *        Only used for unix endpoint in proxy
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] service_instance_key : service-id, instance-id and major-version of the called service-instance
 * @param [in] service_instance : the void service-instance pointer. can convert to specific service-instance
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_register_instance_service(
    void* endpoint, ne_someip_service_instance_spec_t* service_instance_key, const void* service_instance);

/**
 * @brief The unregister instance pointer interface in endpoint used by provided-service-instance side. (called and run in work thread)
 *        Only used for unix endpoint in proxy
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] service_instance_key : service-id, instance-id and major-version of the called service-instance
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t
    ne_someip_endpoint_unregister_instance_service(void* endpoint, ne_someip_service_instance_spec_t* service_instance_key);

/**
 * @brief The register callback function interface in endpoint. Used by all endpoints.
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] callback : the callback function struct
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_register_callback(void* endpoint, ne_someip_endpoint_callback_t* callback);


/**
 * @brief The unregister callback function interface in endpoint. Used by all endpoints.
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] callback : the callback function struct
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unregister_callback(void* endpoint);

/**
 * @brief The function used to registed multi instance for one endpoint. Used by tcp/udp endpoints.
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] instance : the void service-instance pointer
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_reg_inst_end_to_end(void* endpoint, const void* instance);

/**
 * @brief The function used to unregisted multi instance for one endpoint. Used by tcp/udp endpoints.
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] instance : the void service-instance pointer
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_unreg_inst_end_to_end(void* endpoint, const void* instance);

/**
 * @brief Saving the corresponding relation of subscriber event-id and required-service-instance. (called and run in work thread)
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] service_instance_key : the client id of the called service-instance
 * @param [in] event_ids : the subscribed event-id array
 * @param [in] event_cnt : the number of the subscribed event-id array
 * @param [in] service_instance : the void service-instance pointer. can convert to specific service-instance
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
ne_someip_error_code_t ne_someip_endpoint_add_client_subscriber(
    void* endpoint, ne_someip_service_instance_spec_t service_instance_key, ne_someip_event_id_t* event_ids, uint32_t event_cnt,
    void* service_instance);

// 删除client端取消订阅的event_ids相应的接收回调函数
/**
 * @brief Removing the corresponding relation of subscriber event-id and required-service-instance. (called and run in work thread)
 *
 * @param [in] endpoint : the void endpoint pointer, can convert to specific endpoint
 * @param [in] service_instance_key : the client id of the called service-instance
 * @param [in] event_ids : the subscribed event-id array
 * @param [in] event_cnt : the number of the subscribed event-id array
 * @param [in] service_instance : the void service-instance pointer. can convert to specific service-instance
 *
 * @return ne_someip_error_code_ok indicates success, other value indicates failure.
 *
 * @attention Synchronous I/F.
 */
 ne_someip_error_code_t ne_someip_endpoint_remove_client_subscriber(
    void* endpoint, ne_someip_service_instance_spec_t service_instance_key, ne_someip_event_id_t* event_ids, uint32_t event_cnt,
    void* service_instance);

/**
 * @brief get the transmit state of the local unix addr in endpoint
 *
 * @param [in] endpoint : ne_someip_unix_data_endpoint_t pointer.
 * @param [in] local_addr : the local unix domain addr or net addr
 * @param [in] type : the endpoint addr type
 *
 * @return ne_someip_endpoint_transmit_state_not_created : transmit not created
 *         ne_someip_endpoint_transmit_state_startd : transmit can use to normally receive and send data
 *         ne_someip_endpoint_transmit_state_stopped : default state, or transmit was stopped state，can't use to receive and send data
 *         ne_someip_endpoint_transmit_state_prepared : transmit was prepared state，can't use to receive and send data
 *         ne_someip_endpoint_transmit_state_error : transmit error occured，can't use to receive and send data
 *
 * @attention Synchronous I/F.
 */
ne_someip_endpoint_transmit_state_t
    ne_someip_endpoint_transmit_state_get(void* endpoint, void* local_addr, ne_someip_endpoint_addr_type_t type);

ne_someip_endpoint_transmit_link_state_t ne_someip_endpoint_transmit_link_state_get(
    void* endpoint, void* peer_addr, ne_someip_endpoint_addr_type_t type, ne_someip_endpoint_link_role_t role);

uint16_t ne_someip_endpoint_get_port(const void* endpoint);

/****************************callback function*****************************/
void ne_someip_endpoint_link_state_notify(void* endpoint, ne_someip_endpoint_transmit_link_state_t state,
    void* pair_addr);

void ne_someip_endpoint_async_send_reply(void* endpoint, const void* seq_data, ne_someip_error_code_t result);

void ne_someip_endpoint_on_receive(void* endpoint, ne_someip_trans_buffer_struct_t* trans_buffer, void* pair_addr);

/****************************internal function*****************************/
ne_someip_endpoint_base_t* ne_someip_endpoint_base_ref(ne_someip_endpoint_base_t* ep_base);

void ne_someip_endpoint_base_unref(ne_someip_endpoint_base_t* ep_base);

ne_someip_error_code_t ne_someip_endpoint_register_instance_client_ack(void* endpoint,
    ne_someip_endpoint_client_instance_spec_t* client_spec,
    const void* service_instance, ne_someip_endpoint_instance_type_t type);

ne_someip_error_code_t ne_someip_endpoint_unregister_instance_client_ack(void* endpoint,
    ne_someip_endpoint_client_instance_spec_t* client_spec, const void* service_instance, 
    ne_someip_endpoint_instance_type_t type);

ne_someip_error_code_t ne_someip_endpoint_register_instance_service_ack(
    void* endpoint, ne_someip_service_instance_spec_t* service_instance_key, const void* service_instance);

ne_someip_error_code_t
    ne_someip_endpoint_unregister_instance_service_ack(void* endpoint, ne_someip_service_instance_spec_t* service_instance_key);

/********************************** For link *****************************************/
ne_someip_endpoint_transmit_link_info_t* ne_someip_endpoint_find_link_info(
    void* ep, void* peer_addr, ne_someip_endpoint_addr_type_t type, ne_someip_endpoint_link_role_t role);

ne_someip_error_code_t ne_someip_endpoint_delete_link_info(void* ep, void* peer_addr, ne_someip_endpoint_addr_type_t type);

ne_someip_error_code_t ne_someip_endpoint_save_link_info(void* ep, void* peer_addr, ne_someip_endpoint_transmit_link_state_t state,
    ne_someip_endpoint_link_role_t role, ne_someip_transmit_link_t* transmit_link);

ne_someip_transmit_link_t* ne_someip_endpoint_find_transmit_link(void* endpoint, void* peer_addr,
    ne_someip_endpoint_addr_type_t type, ne_someip_endpoint_link_role_t role);

ne_someip_error_code_t ne_someip_endpoint_create_link_client(void* endpoint, void* peer_addr,
    ne_someip_endpoint_addr_type_t type, ne_someip_endpoint_link_role_t role);

ne_someip_error_code_t ne_someip_endpoint_create_link_server(void* endpoint, void* peer_addr,
    ne_someip_endpoint_addr_type_t type, ne_someip_endpoint_link_role_t role);

ne_someip_error_code_t ne_someip_endpoint_destroy_link(
    void* endpoint, void* peer_addr, ne_someip_endpoint_addr_type_t type, ne_someip_endpoint_link_role_t role);
/********************************** For link *****************************************/

ne_someip_error_code_t ne_someip_endpoint_join_group(void* endpoint, ne_someip_endpoint_net_addr_t* interface_addr);

ne_someip_error_code_t ne_someip_endpoint_leave_group(void* endpoint, ne_someip_endpoint_net_addr_t* interface_addr);

int32_t ne_someip_endpoint_get_sync_seq_id(void* ep);

ne_someip_error_code_t ne_someip_endpoint_get_sync_res_code(void* ep, int32_t seq_id);

void ne_someip_endpoint_set_sync_res_code(void* ep, int32_t seq_id, ne_someip_error_code_t code);

#ifdef __cplusplus
}
#endif
#endif // SRC_PROTOCOL_ENDPOINT_NE_SOMEIP_ENDPOINT_COMMON_FUNC_H
/* EOF */