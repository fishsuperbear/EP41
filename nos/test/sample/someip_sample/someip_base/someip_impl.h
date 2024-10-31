#ifndef SOMEIP_IMPL_H_
#define SOMEIP_IMPL_H_

#include <string>

#include "someip/include/cJSON.h"
#include "someip/include/ne_someip_define.h"
#include "someip/include/ne_someip_object.h"
#include "someip/include/ne_someip_looper.h"
#include "someip/include/ne_someip_handler.h"
#include "someip/include/ne_someip_config_define.h"
#include "someip/include/ne_someip_config_parse.h"
#include "someip/include/ne_someip_client_context.h"
#include "someip/include/ne_someip_server_context.h"
#include "someip/include/NESomeIPPayloadDeserializeGeneral.h"
#include "someip/include/NESomeIPPayloadSerializeGeneral.h"

class SomeipImpl {
public:
    SomeipImpl(const std::string& config);
    ~SomeipImpl();

    void Init();
    void Start();
    void Stop();
    void Deinit();

    // server
    void SendRequest();
    void SendEvent();

    // client
    void SendResponse(const ne_someip_header_t* req_header, ne_someip_remote_client_info_t* client_info);

protected:
    static void recv_subscribe_handler_callback(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* list,
        ne_someip_remote_client_info_t* client_info, void* user_data);

    static void send_event_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id,
        ne_someip_error_code_t ret_code, void* user_data);

    static void send_response_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id,
        ne_someip_error_code_t ret_code, void* user_data);

    static void recv_request_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_header_t* header,
        ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);

    static void serivce_status_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status,
        ne_someip_error_code_t ret_code, void* user_data);

    static void someip_find_status_handler_callback(const ne_someip_find_offer_service_spec* spec,
        ne_someip_find_status_t status, ne_someip_error_code_t code, void* user_data);

    static void someip_service_available_handler_callback(const ne_someip_find_offer_service_spec_t* spec,
        ne_someip_service_status_t status, void* user_data);

    static void someip_subscribe_status_handler_callback(ne_someip_required_service_instance_t* instance,
        ne_someip_eventgroup_id_t eventgroup_id, ne_someip_subscribe_status_t status, ne_someip_error_code_t code,
        void* user_data);

    static void someip_send_req_status_handler_callback(ne_someip_required_service_instance_t* instance, void* seq_def,
        ne_someip_method_id_t method_id, ne_someip_error_code_t code, void* user_data);

    static void someip_recv_event_handler_callback(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
        ne_someip_payload_t* payload, void* user_data);

    static void someip_recv_response_handler_callback(ne_someip_required_service_instance_t* instance,
        ne_someip_header_t* header, ne_someip_payload_t* payload, void* user_data);

private:
    int32_t LoadConfig();


private:
    ne_someip_config_t*                     g_someip_config;
    ne_someip_server_context_t*             g_server_context;
    ne_someip_client_context_t*             g_client_context;
    ne_someip_provided_instance_t*          g_provided_instance;
    ne_someip_required_service_instance_t*  g_required_instance;
    ne_someip_find_offer_service_spec_t     g_service_spec;
    std::string                             m_config_file;
};



#endif