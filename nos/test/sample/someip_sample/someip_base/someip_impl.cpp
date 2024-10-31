#include "someip_impl.h"

#include <stdio.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>


void SomeipImpl::recv_subscribe_handler_callback(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* list,
    ne_someip_remote_client_info_t* client_info, void* user_data)
    {
        printf("++++++++++++++server recv_subscribe_handler_callback+++++++++++++++++\n");
        static_cast<SomeipImpl *>(user_data)->SendEvent();
    }

void SomeipImpl::send_event_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id,
    ne_someip_error_code_t ret_code, void* user_data)
    {
        printf("++++++++++++++server send_event_handler_callback+++++++++++++++++\n");

    }

void SomeipImpl::send_response_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id,
    ne_someip_error_code_t ret_code, void* user_data)
    {
        printf("++++++++++++++server send_response_handler_callback+++++++++++++++++\n");
    }

void SomeipImpl::recv_request_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data)
    {
        printf("++++++++++++++server recv_request_handler_callback+++++++++++++++++\n");
        static_cast<SomeipImpl *>(user_data)->SendResponse(header, client_info);
    }

void SomeipImpl::serivce_status_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status,
    ne_someip_error_code_t ret_code, void* user_data)
    {
        printf("++++++++++++++server serivce_status_handler_callback+++++++++++++++++ status: %d\n", status);

    }

void SomeipImpl::someip_find_status_handler_callback(const ne_someip_find_offer_service_spec* spec,
    ne_someip_find_status_t status, ne_someip_error_code_t code, void* user_data)
    {
        printf("++++++++++++++client someip_find_status_handler_callback+++++++++++++++++ status: %d\n", status);

    }

void SomeipImpl::someip_service_available_handler_callback(const ne_someip_find_offer_service_spec_t* spec,
    ne_someip_service_status_t status, void* user_data)
    {
        printf("++++++++++++++client someip_service_available_handler_callback+++++++++++++++++ status: %d\n", status);
        static_cast<SomeipImpl *>(user_data)->SendRequest();
    }

void SomeipImpl::someip_subscribe_status_handler_callback(ne_someip_required_service_instance_t* instance,
    ne_someip_eventgroup_id_t eventgroup_id, ne_someip_subscribe_status_t status, ne_someip_error_code_t code,
    void* user_data)
    {
        printf("++++++++++++++client someip_subscribe_status_handler_callback+++++++++++++++++\n");
    }

void SomeipImpl::someip_send_req_status_handler_callback(ne_someip_required_service_instance_t* instance, void* seq_def,
    ne_someip_method_id_t method_id, ne_someip_error_code_t code, void* user_data)
    {
        printf("++++++++++++++client someip_send_req_status_handler_callback+++++++++++++++++\n");
    }

void SomeipImpl::someip_recv_event_handler_callback(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload, void* user_data)
    {
        printf("++++++++++++++client someip_recv_event_handler_callback+++++++++++++++++\n");
    }

void SomeipImpl::someip_recv_response_handler_callback(ne_someip_required_service_instance_t* instance,
    ne_someip_header_t* header, ne_someip_payload_t* payload, void* user_data)
    {
        printf("++++++++++++++client someip_recv_response_handler_callback+++++++++++++++++\n");
    }

SomeipImpl::SomeipImpl(const std::string& config)
    : g_someip_config(nullptr)
    , g_server_context(nullptr)
    , g_client_context(nullptr)
    , g_provided_instance(nullptr)
    , g_required_instance(nullptr)
    , m_config_file(config)
{
    printf("LoadConfig m_config_file: %s, config: %s\n", m_config_file.c_str(), config.c_str());
    if (LoadConfig() < 0) {
        printf("LoadConfig failed\n");
        g_someip_config = nullptr;
        return;
    }
    g_server_context = ne_someip_server_create(false, 0, 0);
    if (nullptr != g_server_context) {
        g_provided_instance = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config);
    }

    g_service_spec.ins_spec.service_id = g_someip_config->service_array.service_config->service_id;
    g_service_spec.ins_spec.instance_id = g_someip_config->required_instance_array.required_instance_config->instance_id;
    g_service_spec.ins_spec.major_version = g_someip_config->service_array.service_config->major_version;
    g_service_spec.minor_version = g_someip_config->service_array.service_config->minor_version;
    g_client_context = ne_someip_client_context_create(0, false, 0, 0);
    if (nullptr != g_client_context) {
        g_required_instance = ne_someip_client_context_create_req_instance(g_client_context, g_someip_config->required_instance_array.required_instance_config, &g_service_spec.ins_spec);
    }
}

SomeipImpl::~SomeipImpl()
{
    if (nullptr != g_provided_instance) {
        // destroy instance
        ne_someip_server_destroy_instance(g_server_context, g_provided_instance);
        ne_someip_server_unref(g_server_context);
        g_provided_instance = nullptr;
        g_server_context = nullptr;
    }

    if (nullptr != g_required_instance) {
        ne_someip_client_context_destroy_req_instance(g_client_context, g_required_instance);
        ne_someip_client_context_unref(g_client_context);
        g_required_instance = nullptr;
        g_client_context = nullptr;
    }

    ne_someip_config_release_someip_config(&g_someip_config);
}

void SomeipImpl::Init()
{

}

void SomeipImpl::Start()
{
    printf("++++++++++++++Start() begin+++++++++++++++++\n");
    {
        printf("server regsister and offer service\n");
        // server regsister and offer service
        // register hander
        ne_someip_server_reg_subscribe_handler(g_server_context, g_provided_instance, recv_subscribe_handler_callback, this);
        ne_someip_server_reg_event_status_handler(g_server_context, g_provided_instance, send_event_handler_callback, this);
        ne_someip_server_reg_service_status_handler(g_server_context, g_provided_instance, serivce_status_handler_callback, this);
        ne_someip_server_reg_req_handler(g_server_context, g_provided_instance, recv_request_handler_callback, this);
        printf("server register and offer service\n");
        // test offer
        ne_someip_server_offer(g_server_context, g_provided_instance);
    }

    {
        printf("client regsister and find service\n");
        ne_someip_client_context_reg_find_service_handler(g_client_context, &g_service_spec, someip_find_status_handler_callback, this);
        ne_someip_client_context_reg_service_status_handler(g_client_context, &g_service_spec, someip_service_available_handler_callback, this);

        ne_someip_client_context_start_find_service(g_client_context, &g_service_spec, g_someip_config->required_instance_array.required_instance_config);

        ne_someip_client_context_reg_response_handler(g_client_context, g_required_instance, someip_recv_response_handler_callback, this);
        ne_someip_client_context_reg_sub_handler(g_client_context, g_required_instance, someip_subscribe_status_handler_callback, this);
        ne_someip_client_context_reg_send_status_handler(g_client_context, g_required_instance, someip_send_req_status_handler_callback, this);
        ne_someip_client_context_reg_event_handler(g_client_context, g_required_instance, someip_recv_event_handler_callback, this);

        printf("subscribe event group\n");

    }
    printf("++++++++++++++Start() end+++++++++++++++++\n");

}

void SomeipImpl::Deinit()
{

}

void SomeipImpl::Stop()
{
    printf("++++++++++++++Stop() begin+++++++++++++++++\n");
    // server stop
    ne_someip_server_unreg_subscribe_handler(g_server_context, g_provided_instance);
    ne_someip_server_unreg_event_status_handler(g_server_context, g_provided_instance);
    ne_someip_server_unreg_service_status_handler(g_server_context, g_provided_instance);
    ne_someip_server_unreg_req_handler(g_server_context, g_provided_instance);
    ne_someip_server_stop_offer(g_server_context, g_provided_instance);


    // client stop
    ne_someip_client_context_stop_find_service(g_client_context, &g_service_spec);

    ne_someip_client_context_unreg_find_service_handler(g_client_context, &g_service_spec, someip_find_status_handler_callback);
    ne_someip_client_context_unreg_service_status_handler(g_client_context, &g_service_spec, someip_service_available_handler_callback);
    ne_someip_client_context_unreg_response_handler(g_client_context, g_required_instance, someip_recv_response_handler_callback);
    ne_someip_client_context_unreg_sub_handler(g_client_context, g_required_instance, someip_subscribe_status_handler_callback);
    ne_someip_client_context_unreg_send_status_handler(g_client_context, g_required_instance, someip_send_req_status_handler_callback);
    ne_someip_client_context_unreg_event_handler(g_client_context, g_required_instance, someip_recv_event_handler_callback);

}

void SomeipImpl::SendRequest()
{
    printf("++++++++++++++SendRequest() begin+++++++++++++++++\n");
    // send request
    ne_someip_payload_t* payload =  ne_someip_payload_create();
    payload->buffer_list = (ne_someip_payload_slice_t ** )malloc(sizeof(*payload->buffer_list));
    ne_someip_payload_slice_t* slice = (ne_someip_payload_slice_t* )malloc(sizeof(ne_someip_payload_slice_t));
    if (NULL == slice) {
        printf("slice malloc error\n");
        return;
    }
    uint32_t data_len = 2000;
    uint8_t* data = (uint8_t*)malloc(data_len);
    memset(data, 0x00, data_len);
    memcpy(data, "Someip Send Request ------ !!!", strlen("Someip Send Request ------ !!!"));
    slice->data = data;
    slice->length = data_len;
    slice->free_pointer = data;
    *(payload->buffer_list) = slice;
    payload->num = 1;
    ne_someip_header_t header;
    ne_someip_client_context_create_req_header(g_required_instance, &header, g_someip_config->required_instance_array.required_instance_config->method_config_array.method_config_array->method_config->method_id); // method_id is 1

    int32_t ret = ne_someip_client_context_send_request(g_client_context, g_required_instance, NULL, &header, payload);
    if (ne_someip_error_code_ok != ret) {
        printf("client: send method error, ret:[0x%x]\n", ret);
    }
    ne_someip_payload_unref(payload);
    printf("++++++++++++++SendRequest() end+++++++++++++++++\n");
}

void SomeipImpl::SendEvent()
{
    printf("++++++++++++++SendEvent() begin+++++++++++++++++\n");
    // send event
    ne_someip_payload_t* payload =  ne_someip_payload_create();
    payload->buffer_list = (ne_someip_payload_slice_t ** )malloc(sizeof(*payload->buffer_list));
    ne_someip_payload_slice_t* slice = (ne_someip_payload_slice_t* )malloc(sizeof(ne_someip_payload_slice_t));
    if (NULL == slice) {
        printf("slice malloc error\n");
        return;
    }
    uint32_t data_len = 2000;
    uint8_t* data = (uint8_t*)malloc(data_len);
    memset(data, 0x00, data_len);
    memcpy(data, "Someip Send event ~~~~ !!!", strlen("Someip Send event ~~~~ !!!"));
    slice->data = data;
    slice->length = data_len;
    slice->free_pointer = data;
    *(payload->buffer_list) = slice;
    payload->num = 1;
    ne_someip_header_t header;
    ne_someip_server_create_notify_header(g_provided_instance, g_someip_config->provided_instance_array.provided_instance_config->event_config_array.event_config_array->event_config->event_id, &header);

    uint32_t seq_id = 100;
    ne_someip_server_send_event(g_server_context, g_provided_instance, &seq_id, &header, payload);
    ne_someip_payload_unref(payload);
    printf("++++++++++++++SendEvent() end+++++++++++++++++\n");
}

void SomeipImpl::SendResponse(const ne_someip_header_t* req_header, ne_someip_remote_client_info_t* client_info)
{
    printf("++++++++++++++SendResponse() begin+++++++++++++++++\n");
    // send response
    ne_someip_header_t header;
    ne_someip_server_create_resp_header(req_header, &header, ne_someip_message_type_response, 0);

    // 序列化
    ne_someip_payload_t *payload = ne_someip_payload_create();
    if (NULL == payload) {
        printf("ne_someip_payload_create return NULL.\n");
        return;
    }
    NESomeIPExtendAttr attr;
    attr.byte_order = NESomeIPPayloadByteOrder_BE;
    attr.alignment = NESomeIPPayloadAlignment_32;
    NESomeIPPayloadSerializerGeneral serializer(&attr);

    bool value_bool = true;
    int8_t value_int8 = 1;
    int16_t value_int16 = 0x0203;
    int32_t value_int32 = 0x04050607;
    int64_t value_int64 = -2;
    float value_float = 3.14;
    double value_double = 3.1415926;

    serializer.write(value_bool, nullptr);
    serializer.write(value_int8, nullptr);
    serializer.write(value_int16, nullptr);
    serializer.write(value_int32, nullptr);
    serializer.write(value_int64, nullptr);
    serializer.write(value_float, nullptr);
    serializer.write(value_double, nullptr);

    std::vector<uint8_t> buffer = serializer.to_buffer();

    uint8_t *data = (uint8_t *)malloc(buffer.size() * sizeof(uint8_t));
    if (NULL == data) {
        printf("malloc data error.\n");
        free(data);
        free(payload->buffer_list);
        free(payload);
        return;
    }
    memset(data, 0, buffer.size());
    memcpy(data, buffer.data(), buffer.size());
    payload->num = 1;

    // request and response
    header.message_type = ne_someip_message_type_response;
    header.return_code = 0x00;
    if (ne_someip_error_code_ok != ne_someip_server_send_response(g_server_context, g_provided_instance, nullptr, &header, payload, client_info)) {
        printf("[send response] fail, method_id: %d, session_id: %d.", header.method_id, header.session_id);
    }
    ne_someip_payload_unref(payload);
    printf("++++++++++++++SendResponse() end+++++++++++++++++\n");
}

int32_t SomeipImpl::LoadConfig()
{
    printf("++++++++++++++LoadConfig()+++++++++++++++++\n");
    int32_t ret = -1;
    FILE* file_fd = fopen(m_config_file.c_str(), "rb");
    if (NULL == file_fd) {
        printf("open file %s failed\n", m_config_file.c_str());
        return ret;
    }

    fseek(file_fd, 0, SEEK_END);
    int64_t data_length = ftell(file_fd);
    if (data_length <= 0) {
        printf("ftell %s failed errno:%d errmsg:%s\n", m_config_file.c_str(), errno, strerror(errno));
        fclose(file_fd);
        return ret;
    }
    printf("config: %s data_length:[%ld]\n", m_config_file.c_str(), data_length);
    rewind(file_fd);

    char* file_data = (char*)malloc(data_length + 1);
    if (NULL == file_data) {
        printf("file_data malloc error\n");
        fclose(file_fd);
        return ret;
    }
    memset(file_data, 0, data_length + 1);

    fread(file_data, 1, data_length, file_fd);
    fclose(file_fd);

    cJSON* config_object = cJSON_Parse(file_data);
    // ne_someip_log_info("config:\n %s", cJSON_Print(config_object));
    if (NULL != file_data) {
        free(file_data);
        file_data = NULL;
    }
    g_someip_config = ne_someip_config_parse_someip_config(config_object);
    printf("parse config successful:\n");
    printf("if_name: %s\n", g_someip_config->network_array.network_config->if_name);
    printf("ip_addr: %X\n", g_someip_config->network_array.network_config->ip_addr);
    printf("multicast_ip: %X\n", g_someip_config->network_array.network_config->multicast_ip);
    printf("multicast_port: %d\n", g_someip_config->network_array.network_config->multicast_port);
    printf("service_name: %s\n", g_someip_config->service_array.service_config->short_name);
    printf("service_id: %d\n", g_someip_config->service_array.service_config->service_id);
    printf("major_version: %d\n", g_someip_config->service_array.service_config->major_version);
    printf("minor_version: %d\n", g_someip_config->service_array.service_config->minor_version);
    printf("provider service_name: %s\n", g_someip_config->provided_instance_array.provided_instance_config->short_name);
    printf("provider service_id: %d\n", g_someip_config->provided_instance_array.provided_instance_config->instance_id);
    ret = 1;
    return ret;
}
