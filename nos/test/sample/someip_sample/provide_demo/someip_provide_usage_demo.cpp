#include <stdio.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>


#include "someip/include/cJSON.h"
#include "someip/include/ne_someip_client_context.h"
#include "someip/include/ne_someip_config_define.h"
#include "someip/include/ne_someip_define.h"
#include "someip/include/ne_someip_object.h"
#include "someip/include/ne_someip_looper.h"
#include "someip/include/ne_someip_handler.h"
#include "someip/include/ne_someip_server_context.h"
#include "someip/include/ne_someip_config_parse.h"
#include "someip/include/NESomeIPPayloadDeserializeGeneral.h"
#include "someip/include/NESomeIPPayloadSerializeGeneral.h"


#define COMMAND_SIZE 20

#define LOG(format, ...) \
 { \
    char print_msg[1024]= { 0 };    \
    struct timeval tv;              \
    gettimeofday(&tv, NULL);     \
    struct tm *timeinfo = localtime(&tv.tv_sec);        \
    uint32_t milliseconds = tv.tv_usec / 1000;          \
    char time_buf[64] = { 0 };                          \
    memset(time_buf, 0x00, sizeof(time_buf));           \
    memset(print_msg, 0x00, sizeof(print_msg));         \
    snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d ",        \
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,             \
        timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, milliseconds);          \
    snprintf(print_msg, sizeof(print_msg), (format), ##__VA_ARGS__);                   \
    printf("[%s] [%d %ld %s@%s(%d) | %s]\n", time_buf, getpid(), syscall(__NR_gettid), \
        __FUNCTION__, (NULL == strrchr(__FILE__, '/')) ? __FILE__: (strrchr(__FILE__, '/') + 1), __LINE__, (print_msg)); \
 }


/* ================== global values ================== */
ne_someip_server_context_t* g_server_context = NULL;
ne_someip_provided_instance_t* g_pro_instance;
ne_someip_config_t* g_someip_config;

/* ================== test case functions ================== */
static void test_send_multicast(void);
static void test_offer_service(void);

/* ================== command declaration ================== */
void help_message(void) {
    printf("======================================\n");
    printf(" * 0-test exit\n");
    printf(" * 1-test send multicast\n");
    printf(" * 2-test offer service\n");
    printf(" * 3-test send event\n");
    printf(" * 4-test stop offer\n");
    printf(" * 5-test send response\n");
    printf("======================================\n");
}
/* ================== command functions ================== */
ne_someip_config_t* load_config(const char* someip_config_path);

void test_offer_service(void);
void test_send_event(void);
void test_stop_offer_service(void);
void test_send_udp_tp(void);


void send_response_func(ne_someip_provided_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload, const ne_someip_remote_client_info_t* client_info);

/* ================== register callback functions ================== */
void recv_subscribe_handler_callback(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* list,
    ne_someip_remote_client_info_t* client_info, void* user_data);
void send_event_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id,
    ne_someip_error_code_t ret_code, void* user_data);
void send_response_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id,
    ne_someip_error_code_t ret_code, void* user_data);
void recv_request_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
void serivce_status_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status,
    ne_someip_error_code_t ret_code, void* user_data);


/* ================== main function ================== */
int main(int argc, char** argv) {
    /* ================== someip config load ================== */
    const char* someip_config_path = "conf/someip_config.json";
    load_config(someip_config_path);
    if (NULL == g_someip_config) {
        printf("load confog error");
        return 0;
    }
    /* ================== create send event monitor thread ================== */

    char command[COMMAND_SIZE];
    while (1) {
        help_message();
        memset(command, 0x00, COMMAND_SIZE);
        scanf("%s", command);
        int command_type = atoi(&command[0]);
        if (command_type == 0) {
            break;
        } else if (command_type == 1) {
            test_send_multicast();
        } else if (command_type == 2) {
            test_offer_service();
        } else if (command_type == 3) {
            test_send_event();
        } else if (command_type == 4) {
            test_stop_offer_service();
        } else if (command_type == 5) {
            // test_send_udp_tp();
        } else {
            help_message();
        }
    }
    return 0;
}

/* ================== callback functions ================== */
void recv_subscribe_handler_callback(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* list,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    printf("\n++++++++++++++test_recv_subscribe_handler+++++++++++++++++");
}

void send_event_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id,
    ne_someip_error_code_t ret_code, void* user_data) {
    printf("\n++++++++++++++test_send_event_handler+++++++++++++++++");
}

void send_response_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id,
    ne_someip_error_code_t ret_code, void* user_data) {
    printf("\n++++++++++++++test_send_response_handler+++++++++++++++++");
}

void recv_request_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data) {
    printf("\n++++++++++++++test_recv_request_handler+++++++++++++++++");

    std::vector<uint8_t> buffer (payload->buffer_list[0]->data, payload->buffer_list[0]->data + payload->buffer_list[0]->length);
    NESomeIPExtendAttr attr;
    attr.byte_order = NESomeIPPayloadByteOrder_BE;
    attr.alignment = NESomeIPPayloadAlignment_32;
    NESomeIPPayloadDeserializerGeneral deserializer(&attr);
    deserializer.from_buffer(buffer);


    bool value_bool;
    int8_t value_int8;
    int16_t value_int16;
    int32_t value_int32;
    int64_t value_int64;
    float value_float;
    double value_double;
    deserializer.read(value_bool, nullptr);
    deserializer.read(value_int8, nullptr);
    deserializer.read(value_int16, nullptr);
    deserializer.read(value_int32, nullptr);
    deserializer.read(value_int64, nullptr);
    deserializer.read(value_float, nullptr);
    deserializer.read(value_double, nullptr);

    printf("value_bool = [%d]\n", value_bool);
    printf("value_int8 = [0x%02x]\n", value_int8);
    printf("value_int16 = [0x%04x]\n", value_int16);
    printf("value_int32 = [0x%08x]\n", value_int32);
    printf("value_int64 = [%ld]\n", value_int64);
    printf("value_float = [%.2f]\n", value_float);
    printf("value_double = [%.8lf]\n", value_double);

    printf("is_empty = %d\n", deserializer.is_empty());
    printf("last_error:[%d]\n", deserializer.get_last_error());

    // send response
    ne_someip_payload_t* tmp_payload =  ne_someip_payload_create();
    tmp_payload->buffer_list = (ne_someip_payload_slice_t ** )malloc(sizeof(*tmp_payload->buffer_list));
    ne_someip_payload_slice_t* slice = (ne_someip_payload_slice_t* )malloc(sizeof(ne_someip_payload_slice_t));
    if (NULL == slice) {
        printf("\nslice malloc error");
        return;
    }
    uint8_t* data1 = (uint8_t*)malloc(11);
    memset(data1, 1, 10);
    slice->data = data1;
    slice->length = 11;
    slice->free_pointer = data1;
    *(tmp_payload->buffer_list) = slice;
    tmp_payload->num = 1;
    ne_someip_header_t tmp_header;
    ne_someip_server_create_resp_header(header, &tmp_header, ne_someip_message_type_response, 0);

    send_response_func(instance, &tmp_header, tmp_payload, client_info);
}

void send_response_func(ne_someip_provided_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload_, const ne_someip_remote_client_info_t* client_info) {
    printf("\n++++++++++++++++++send response+++++++++++++++++++++");
    if (1 == header->method_id)  {

        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        if (ne_someip_error_code_ok != ne_someip_server_send_response(g_server_context, g_pro_instance, nullptr, header, payload_, client_info)) {
            printf("[send response] fail, method_id: %d, session_id: %d.", header->method_id, header->session_id);
        }
    }

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

    ne_someip_payload_unref(payload_);
    ne_someip_payload_unref(payload);
}

void serivce_status_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status,
    ne_someip_error_code_t ret_code, void* user_data) {
    printf("\n++++++++++++++++++service status call back status [%d]+++++++++++++++++++++\n", status);
    ne_someip_service_instance_spec_t ins_spec;
    bool ret = ne_someip_server_get_instance_info(instance, &ins_spec);
    if (ret) {
        printf("\nservice status call back status [%d], service id[%d], instance id[%d], major version[%d]\n",
            status, ins_spec.service_id, ins_spec.instance_id, ins_spec.major_version);
    }
}

/* ================== command functions ================== */
ne_someip_config_t* load_config(const char* someip_config_path) {
    FILE* file_fd = fopen(someip_config_path, "rb");
    if (NULL == file_fd) {
        printf("\nopen file failed");
        return NULL;
    }

    fseek(file_fd, 0, SEEK_END);
    int64_t data_length = ftell(file_fd);
    if (data_length <= 0) {
        printf("\nftell failed errno:%d errmsg:%s", errno, strerror(errno));
        fclose(file_fd);
        return NULL;
    }
    printf("\ndata_length:[%ld]", data_length);
    rewind(file_fd);

    char* file_data = (char*)malloc(data_length + 1);
    if (NULL == file_data) {
        printf("file_data malloc error\n");
        fclose(file_fd);
        return NULL;
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
    return g_someip_config;
}

/* ================== test case functions ================== */
void test_send_event(void) {
    printf("test_send_event\n");
    ne_someip_sequence_id_t* seq_id = (ne_someip_sequence_id_t*)malloc(sizeof(ne_someip_sequence_id_t));
    *seq_id = 11;
    // instance1, eventgroup 1001
    ne_someip_payload_t *payload = ne_someip_payload_create();
    if (NULL == payload)
    {
        printf("ne_someip_payload_create return NULL.\n");
        return;
    }
    payload->buffer_list = (ne_someip_payload_slice_t **)malloc(sizeof(ne_someip_payload_slice_t));
    if (NULL == payload->buffer_list)
    {
        printf("malloc error.\n");
        free(payload);
        return;
    }
    ne_someip_payload_slice_t *payload_slice = (ne_someip_payload_slice_t *)malloc(sizeof(ne_someip_payload_slice_t));
    if (NULL == payload_slice)
    {
        printf("malloc payload_slice error.\n");
        free(payload->buffer_list);
        free(payload);
        return;
    }

    // 序列化
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
    if (NULL == data)
    {
        printf("malloc data error.\n");
        free(data);
        free(payload->buffer_list);
        free(payload);
        return;
    }
    memset(data, 0, buffer.size());
    memcpy(data, buffer.data(), buffer.size());
    payload_slice->free_pointer = data;
    payload_slice->data = data;
    payload_slice->length = buffer.size();

    payload->buffer_list[0] = payload_slice;
    payload->num = 1;

    ne_someip_header_t header1;
    ne_someip_server_create_notify_header(g_pro_instance, 1001, &header1);
    ne_someip_payload_ref(payload);
    ne_someip_server_send_event(g_server_context, g_pro_instance, seq_id, &header1, payload);
    sleep(1);
}

void test_send_multicast(void)
{
    printf("test_send_multicast\n");
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        printf("Failed to create socket, errno: %s\n", strerror(errno));
        return;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g_someip_config->network_array.network_config->multicast_port);  // 组播端口号
    addr.sin_addr.s_addr = g_someip_config->network_array.network_config->multicast_ip;
    // inet_pton(AF_INET, "224.224.224.244", &addr.sin_addr);  // 组播地址

    const char* message = "Hello, multicast!";
    uint32_t retry = 0;
    while (retry < 10) {
        if (sendto(sockfd, message, strlen(message), 0, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            printf("Failed to send message, errno: %s\n", strerror(errno));
            break;
        }
        printf("Send message, len: %ld, data: %s!\n", strlen(message), message);
        ++retry;
        sleep(3);
    }

    close(sockfd);
}

void test_offer_service(void)
{
    printf("test_offer_service\n");
    // create server context
    g_server_context = ne_someip_server_create(false, 0, 0);
    if (NULL == g_server_context) {
        printf("create server failed\n");
        return;
    }
    // init config
    if (NULL == g_someip_config) {
        printf("load config error\n");
        return;
    }
    // create instance
    g_pro_instance = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config);
    if (NULL == g_pro_instance) {
        printf("create p_instance falied\n");
        return;
    }
    printf("test_offer_service register hander\n");
    // register hander
    uint8_t* user_data = (uint8_t*)malloc(sizeof(10));
    ne_someip_server_reg_subscribe_handler(g_server_context, g_pro_instance, recv_subscribe_handler_callback, user_data);
    ne_someip_server_reg_event_status_handler(g_server_context, g_pro_instance, send_event_handler_callback, user_data);
    ne_someip_server_reg_service_status_handler(g_server_context, g_pro_instance, serivce_status_handler_callback, user_data);
    ne_someip_server_reg_req_handler(g_server_context, g_pro_instance, recv_request_handler_callback, user_data);
    printf("test_offer_service offer\n");
    // test offer
    ne_someip_server_offer(g_server_context, g_pro_instance);
}


void test_stop_offer_service()
{
    printf("test_stop_offer_service\n");
    ne_someip_server_unreg_subscribe_handler(g_server_context, g_pro_instance);
    ne_someip_server_unreg_event_status_handler(g_server_context, g_pro_instance);
    ne_someip_server_unreg_service_status_handler(g_server_context, g_pro_instance);
    ne_someip_server_unreg_req_handler(g_server_context, g_pro_instance);
    ne_someip_server_stop_offer(g_server_context, g_pro_instance);
    // destroy instance
    ne_someip_server_destroy_instance(g_server_context, g_pro_instance);
    ne_someip_server_unref(g_server_context);
}


void test_send_udp_tp()
{
    uint32_t retry_count = 0;
    while (retry_count < 10)
    {
        {
            ne_someip_payload_t *payload1 = ne_someip_payload_create();
            ne_someip_payload_slice_t *slice1 = (ne_someip_payload_slice_t *)malloc(sizeof(ne_someip_payload_slice_t));

            uint8_t *data1 = (uint8_t *)malloc(1500);
            uint8_t aa1[1500] = "123456789";
            aa1[700] = 97;
            aa1[1490] = 98;
            aa1[1499] = 99;
            memcpy(data1, aa1, 1500);
            slice1->data = data1;
            slice1->length = 1500;
            slice1->free_pointer = data1;
            payload1->buffer_list = (ne_someip_payload_slice_t **)malloc(sizeof(ne_someip_payload_slice_t *));
            *(payload1->buffer_list) = slice1;
            payload1->num = 1;
            ne_someip_header_t header1;
            ne_someip_server_create_notify_header(g_pro_instance, 1001, &header1);
            auto status = ne_someip_server_send_event(g_server_context, g_pro_instance, NULL, &header1, payload1);
            if (ne_someip_error_code_ok != status) {
                printf("send event id 1001 fail");
            }
            ne_someip_payload_unref(payload1);
        }
        sleep(1);
        { // 1003
            ne_someip_payload_t *payload1 = ne_someip_payload_create();
            ne_someip_payload_slice_t *slice1 = (ne_someip_payload_slice_t *)malloc(sizeof(ne_someip_payload_slice_t));

            uint8_t *data1 = (uint8_t *)malloc(1500);
            uint8_t aa1[1500] = "123456789";
            aa1[700] = 97;
            aa1[1490] = 98;
            aa1[1499] = 99;
            memcpy(data1, aa1, 1500);
            slice1->data = data1;
            slice1->length = 1500;
            slice1->free_pointer = data1;
            payload1->buffer_list = (ne_someip_payload_slice_t **)malloc(sizeof(ne_someip_payload_slice_t *));
            *(payload1->buffer_list) = slice1;
            payload1->num = 1;

            ne_someip_header_t header1;
            ne_someip_server_create_notify_header(g_pro_instance, 1003, &header1);
            auto status = ne_someip_server_send_event(g_server_context, g_pro_instance, NULL, &header1, payload1);
            if (ne_someip_error_code_ok != status) {
                printf("send event id 1003 fail");
            }
            ne_someip_payload_unref(payload1);
        }
        ++retry_count;
    }
}