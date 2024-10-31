#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <errno.h>
#include <string>

#include "someip/include/cJSON.h"
#include "someip/include/ne_someip_define.h"
#include "someip/include/ne_someip_config_define.h"
#include "someip/include/ne_someip_config_parse.h"
#include "someip/include/ne_someip_client_context.h"
#include "someip/include/ne_someip_server_context.h"
#include "someip/include/ne_someip_handler.h"
#include "someip/include/ne_someip_looper.h"
#include "someip/include/ne_someip_object.h"
#include "someip/include/NESomeIPPayloadDeserializeGeneral.h"
#include "someip/include/NESomeIPPayloadSerializeGeneral.h"

#define COMMAND_SIZE 20

#define LOG(format, ...) \
 { \
    char print_msg[1024]= { 0 };    \
    struct timeval tv;              \
    gettimeofday(&tv, nullptr);     \
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
        __FUNCTION__, (nullptr == strrchr(__FILE__, '/')) ? __FILE__: (strrchr(__FILE__, '/') + 1), __LINE__, (print_msg)); \
 }


/* ================== global values ================== */
static ne_someip_client_context_t* g_client_context = NULL;
static ne_someip_config_t* g_someip_config = NULL;
static ne_someip_required_service_instance_config_t* g_req_config = NULL;
static ne_someip_find_offer_service_spec_t g_service_spec;
static ne_someip_required_service_instance_t* g_req_instance;

/* ================== callback functions ================== */
static void client_print_recv_message(ne_someip_header_t* header, ne_someip_payload_t* payload);
static void someip_find_status_handler_callback(const ne_someip_find_offer_service_spec_t* spec,
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

/* ================== command functions ================== */
static ne_someip_config_t* load_config();

/* ================== test case functions ================== */
static void test_recv_multicast(void);
static void test_find_and_subscribe(void);
static void test_send_and_recv(void);

/* ================== command functions ================== */
static void help_message(void) {
    printf("====================================\n");
    printf("0 test exit\n");
    printf("1 test recv multicast\n");
    printf("2 test find subscribe\n");
    printf("3 test send recv\n");
    printf("====================================\n");
}
/* ================== main function ================== */
int main(int argc, char** argv) {
    char command[COMMAND_SIZE];

    while (1) {
        help_message();
        memset(command, 0x00, COMMAND_SIZE);
        scanf("%s", command);
        int command_type = atoi(&command[0]);
        if (command_type == 0) {
            break;
        } else if (command_type == 1) {
            test_recv_multicast();
        } else if (command_type == 2) {
            test_find_and_subscribe();
        } else if (command_type == 3) {
            test_send_and_recv();
        } else {
            help_message();
        }
    }

    return 0;
}
/* ================== callback functions ================== */
void client_print_recv_message(ne_someip_header_t* header, ne_someip_payload_t* payload) {
    printf("client_print_recv_message.\n");
    if (NULL == header) {
        printf("header is NULL.\n");
        return;
    }

    printf("+++++++++++++++ service_id [%d] +++++++++++++++++\n", header->service_id);
    printf("+++++++++++++++ method_id [%d] +++++++++++++++++\n", header->method_id);
    printf("+++++++++++++++ message_length [%d] +++++++++++++++++\n", header->message_length);
    printf("+++++++++++++++ client_id [%d] +++++++++++++++++\n", header->client_id);
    printf("+++++++++++++++ session_id [%d] +++++++++++++++++\n", header->session_id);
    printf("+++++++++++++++ protocol_version [%d] +++++++++++++++++\n", header->protocol_version);
    printf("+++++++++++++++ interface_version [%d] +++++++++++++++++\n", header->interface_version);
    printf("+++++++++++++++ message_type [%d] +++++++++++++++++\n", header->message_type);
    printf("+++++++++++++++ return_code [%d] +++++++++++++++++\n", header->return_code);

    // if(header->method_id == 1)
    // {
        // 反序列化
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
    // }
}

void someip_find_status_handler_callback(const ne_someip_find_offer_service_spec* spec,
    ne_someip_find_status_t status, ne_someip_error_code_t code, void* user_data) {
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("++++++++++ someip_find_status_handler_callback +++++++++++++++\n");
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\n");
}

void someip_service_available_handler_callback(const ne_someip_find_offer_service_spec_t* spec,
    ne_someip_service_status_t status, void* user_data) {
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("++++++++++ someip_service_available_handler_callback +++++++++++++++\n");
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\n");
}

void someip_subscribe_status_handler_callback(ne_someip_required_service_instance_t* instance,
    ne_someip_eventgroup_id_t eventgroup_id, ne_someip_subscribe_status_t status, ne_someip_error_code_t code,
    void* user_data) {
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("++++++++++ someip_subscribe_status_handler_callback status [%d]+++++++++++++++\n", status);
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\n");
}

void someip_send_req_status_handler_callback(ne_someip_required_service_instance_t* instance, void* seq_def,
    ne_someip_method_id_t method_id, ne_someip_error_code_t code, void* user_data) {
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("++++++++++ test_someip_send_req_status_handler +++++++++++++++\n");
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\n");
}

void someip_recv_event_handler_callback(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload, void* user_data) {
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("++++++++++ test_someip_recv_event_handler start +++++++++++++++\n");

    client_print_recv_message(header, payload);

    printf("++++++++++ test_someip_recv_event_handler stop +++++++++++++++\n");
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\n");
}

void someip_recv_response_handler_callback(ne_someip_required_service_instance_t* instance,
    ne_someip_header_t* header, ne_someip_payload_t* payload, void* user_data) {
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("++++++++++ test_someip_recv_resp_handler start +++++++++++++++\n");

    client_print_recv_message(header, payload);

    printf("++++++++++ test_someip_recv_resp_handler stop +++++++++++++++\n");
    printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf("\n");
}
/* ================== command functions ================== */
ne_someip_config_t* load_config() {
    printf("load_config.\n");
    const char* someip_config_path = "conf/someip_config.json";
    // fopen file
    FILE* file_fd = fopen(someip_config_path, "rb");
    if (NULL == file_fd) {
        printf("open file failed");
        return NULL;
    }

    fseek(file_fd, 0, SEEK_END);
    long data_length = ftell(file_fd);
    if (data_length <= 0) {
        printf("ftell failed errno:%d errmsg:%s", errno, strerror(errno));
        fclose(file_fd);
        return NULL;
    }
    printf("data_length:[%lu]", data_length);
    rewind(file_fd);

    char* file_data = (char*)malloc(data_length + 1);
    if (NULL == file_data) {
        printf("file_data malloc error");
        fclose(file_fd);
        return NULL;
    }
    memset(file_data, 0, data_length + 1);

    fread(file_data, 1, data_length, file_fd);
    fclose(file_fd);

    cJSON* config_object = cJSON_Parse(file_data);
    if (NULL != file_data) {
        free(file_data);
        file_data = NULL;
    }

    // parse someip config
    g_someip_config = ne_someip_config_parse_someip_config(config_object);
    if (NULL == g_someip_config) {
        printf("parse config failed.\n");
        return NULL;
    }
    g_req_config = g_someip_config->required_instance_array.required_instance_config;
    printf("parse config successful:\n");
    printf("if_name: %s\n", g_someip_config->network_array.network_config->if_name);
    printf("ip_addr: %x\n", g_someip_config->network_array.network_config->ip_addr);
    printf("multicast_ip: %X\n", g_someip_config->network_array.network_config->multicast_ip);
    printf("multicast_port: %d\n", g_someip_config->network_array.network_config->multicast_port);
    printf("service_name: %s\n", g_someip_config->service_array.service_config->short_name);
    printf("service_id: %d\n", g_someip_config->service_array.service_config->service_id);
    printf("major_version: %d\n", g_someip_config->service_array.service_config->major_version);
    printf("minor_version: %d\n", g_someip_config->service_array.service_config->minor_version);
    printf("require service_name: %s\n", g_someip_config->required_instance_array.required_instance_config->short_name);
    printf("require service_id: %d\n", g_someip_config->required_instance_array.required_instance_config->instance_id);
    return g_someip_config;
}


/* ================== test case functions ================== */
void test_recv_multicast(void)
{
    printf("test_recv_multicast.\n");
    load_config();
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        printf("Failed to create socket, errno: %s\n", strerror(errno));
        return;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g_someip_config->network_array.network_config->multicast_port);  // 组播端口号
    addr.sin_addr.s_addr = g_someip_config->network_array.network_config->multicast_ip;  // 组播地址

    // inet_pton(AF_INET, "224.224.224.244", &addr.sin_addr);  // 组播地址

    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("Failed to bind socket, errno: %s\n", strerror(errno));
        close(sockfd);
        return;
    }

    struct ip_mreq mreq;
    memset(&mreq, 0, sizeof(mreq));
    mreq.imr_multiaddr.s_addr = g_someip_config->network_array.network_config->multicast_ip;  // 组播地址
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);  // 本地接口地址

    if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        printf("Failed to join multicast group, errno: %s\n", strerror(errno));
        close(sockfd);
        return;
    }

    uint32_t retry = 0;
    while (retry < 10) {
        char buf[1024];
        memset(buf, 0, sizeof(buf));
        socklen_t addrlen = sizeof(addr);

        int32_t len = recvfrom(sockfd, buf, sizeof(buf), 0, (struct sockaddr*)&addr, &addrlen);
        if (len < 0) {
            printf("Failed to receive message, errno: %s\n", strerror(errno));
            break;
        }
        printf("Received message len: %d, data: %s\n", len, buf);
        if (std::string("Hello, multicast!") == std::string(buf)) {
            break;
        }
        ++retry;
        sleep(3);
    }

    close(sockfd);
}


void test_find_and_subscribe(void)
{
    printf("test_find_and_subscribe.\n");
    load_config();
    ne_someip_error_code_t res = ne_someip_error_code_unknown;
    g_client_context = ne_someip_client_context_create(0, false, 0, 0);
    g_service_spec.ins_spec.service_id = g_someip_config->service_array.service_config->service_id;
    g_service_spec.ins_spec.instance_id = g_someip_config->required_instance_array.required_instance_config->instance_id;
    g_service_spec.ins_spec.major_version = g_someip_config->service_array.service_config->major_version;
    g_service_spec.minor_version = g_someip_config->service_array.service_config->minor_version;

    printf("start client_find.\n");
    ne_someip_client_context_reg_find_service_handler(g_client_context, &g_service_spec, someip_find_status_handler_callback, NULL);
    ne_someip_client_context_reg_service_status_handler(g_client_context, &g_service_spec, someip_service_available_handler_callback, NULL);

    g_req_config = g_someip_config->required_instance_array.required_instance_config;

    ne_someip_find_local_offer_services_t* local_services = ne_someip_client_create_find_local_services();
    res = ne_someip_client_context_query_offered_instances(g_client_context, &g_service_spec, local_services);
    printf("local_services->local_services_num is %d, res: %d\n", local_services->local_services_num, res);
    for (uint32_t i = 0; i < local_services->local_services_num; ++i) {
        printf("find local service: service_id [%d], instance_id [%d], major_version [%d], minor_version [%d]\n",
            local_services->local_services->ins_spec.service_id, local_services->local_services->ins_spec.instance_id,\
            local_services->local_services->ins_spec.major_version, local_services->local_services->minor_version);
    }
    ne_someip_client_destroy_find_local_services(local_services);


    g_req_instance = ne_someip_client_context_create_req_instance(g_client_context, g_req_config, &g_service_spec.ins_spec);


    printf("start client_subscribe.\n");
    res = ne_someip_client_context_reg_response_handler(g_client_context, g_req_instance, someip_recv_response_handler_callback, NULL);
    res = ne_someip_client_context_reg_sub_handler(g_client_context, g_req_instance, someip_subscribe_status_handler_callback, NULL);
    res = ne_someip_client_context_reg_send_status_handler(g_client_context, g_req_instance, someip_send_req_status_handler_callback, NULL);
    res = ne_someip_client_context_reg_event_handler(g_client_context, g_req_instance, someip_recv_event_handler_callback, NULL);

    ne_someip_client_context_start_subscribe_eventgroup(g_client_context, g_req_instance, 3536);
    ne_someip_client_context_start_subscribe_eventgroup(g_client_context, g_req_instance, 5021);


    printf("start client_stop_find.\n");
    ne_someip_client_context_stop_find_service(g_client_context, &g_service_spec);


    printf("start client_unsubscribe.\n");
    ne_someip_client_context_unreg_find_service_handler(g_client_context, &g_service_spec, someip_find_status_handler_callback);
    ne_someip_client_context_unreg_service_status_handler(g_client_context, &g_service_spec, someip_service_available_handler_callback);
    ne_someip_client_context_unreg_response_handler(g_client_context, g_req_instance, someip_recv_response_handler_callback);
    ne_someip_client_context_unreg_sub_handler(g_client_context, g_req_instance, someip_subscribe_status_handler_callback);
    ne_someip_client_context_unreg_send_status_handler(g_client_context, g_req_instance, someip_send_req_status_handler_callback);
    ne_someip_client_context_unreg_event_handler(g_client_context, g_req_instance, someip_recv_event_handler_callback);

    printf("start wait unreg completed for 5s.\n");
    sleep(5);
    ne_someip_client_context_destroy_req_instance(g_client_context, g_req_instance);
    ne_someip_client_context_unref(g_client_context);
    ne_someip_config_release_someip_config(&g_someip_config);


}


void test_send_and_recv()
{
    printf("test_send_and_recv.\n");
    load_config();
    ne_someip_error_code_t res = ne_someip_error_code_unknown;
    g_client_context = ne_someip_client_context_create(0, false, 0, 0);
    g_service_spec.ins_spec.service_id = g_someip_config->service_array.service_config->service_id;
    g_service_spec.ins_spec.instance_id = g_someip_config->required_instance_array.required_instance_config->instance_id;
    g_service_spec.ins_spec.major_version = g_someip_config->service_array.service_config->major_version;
    g_service_spec.minor_version = g_someip_config->service_array.service_config->minor_version;

    printf("start client_find.\n");
    ne_someip_client_context_reg_find_service_handler(g_client_context, &g_service_spec, someip_find_status_handler_callback, NULL);
    ne_someip_client_context_reg_service_status_handler(g_client_context, &g_service_spec, someip_service_available_handler_callback, NULL);

    g_req_config = g_someip_config->required_instance_array.required_instance_config;
    ne_someip_client_context_start_find_service(g_client_context, &g_service_spec, g_req_config);

    ne_someip_find_local_offer_services_t* local_services = ne_someip_client_create_find_local_services();
    res = ne_someip_client_context_query_offered_instances(g_client_context, &g_service_spec, local_services);
    printf("local_services->local_services_num is %d, res: %d\n", local_services->local_services_num, res);
    for (uint32_t i = 0; i < local_services->local_services_num; ++i) {
        printf("find local service: service_id [%d], instance_id [%d], major_version [%d], minor_version [%d]\n",
            local_services->local_services->ins_spec.service_id, local_services->local_services->ins_spec.instance_id,\
            local_services->local_services->ins_spec.major_version, local_services->local_services->minor_version);
    }
    ne_someip_client_destroy_find_local_services(local_services);

    g_req_instance = ne_someip_client_context_create_req_instance(g_client_context, g_req_config, &g_service_spec.ins_spec);

    printf("start client_subscribe.\n");
    res = ne_someip_client_context_reg_response_handler(g_client_context, g_req_instance, someip_recv_response_handler_callback, NULL);
    res = ne_someip_client_context_reg_sub_handler(g_client_context, g_req_instance, someip_subscribe_status_handler_callback, NULL);
    res = ne_someip_client_context_reg_send_status_handler(g_client_context, g_req_instance, someip_send_req_status_handler_callback, NULL);
    res = ne_someip_client_context_reg_event_handler(g_client_context, g_req_instance, someip_recv_event_handler_callback, NULL);

    ne_someip_client_context_start_subscribe_eventgroup(g_client_context, g_req_instance, 3536);
    ne_someip_client_context_start_subscribe_eventgroup(g_client_context, g_req_instance, 5021);

    printf("======start send_udp_tp begin.=====\n");
    uint32_t  retry_count = 0;
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

            ne_someip_header_t header;
            res = ne_someip_client_context_create_req_header(g_req_instance, &header, 1); // method_id is 1

            res = ne_someip_client_context_send_request(g_client_context, g_req_instance, NULL, &header, payload1);
            ne_someip_payload_unref(payload1);
        }
        sleep(1);

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

            ne_someip_header_t header;
            res = ne_someip_client_context_create_req_header(g_req_instance, &header, 2); // field method_id is 2

            res = ne_someip_client_context_send_request(g_client_context, g_req_instance, NULL, &header, payload1);
            ne_someip_payload_unref(payload1);
        }

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

            ne_someip_header_t header;
            res = ne_someip_client_context_create_req_header(g_req_instance, &header, 7488); // field method_id is 2

            res = ne_someip_client_context_send_request(g_client_context, g_req_instance, NULL, &header, payload1);
            ne_someip_payload_unref(payload1);
        }
        ++retry_count;
    }

    printf("======start send_udp_tp end.=====\n");

    printf("~~~~~start send_serialize end.~~~~~\n");
    printf("serialize sample common data.\n");
    retry_count = 0;
    while (retry_count < 10) {
        ne_someip_header_t header;
        res = ne_someip_client_context_create_req_header(g_req_instance, &header, 1);  // method_id is 1

        // someip payload 创建
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

        res = ne_someip_client_context_send_request(g_client_context, g_req_instance, NULL, &header, payload);
        ne_someip_payload_unref(payload);
        ++retry_count;
        sleep(1);

    }
    //  test long data
    printf("serialize object long data.\n");
    retry_count = 0;
    while (retry_count < 10) {
        ne_someip_header_t header;
        res = ne_someip_client_context_create_req_header(g_req_instance, &header, 1);  // method_id is 1

        // someip payload 创建
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

        res = ne_someip_client_context_send_request(g_client_context, g_req_instance, NULL, &header, payload);
        ne_someip_payload_unref(payload);
        ++retry_count;
        sleep(1);
    }

    printf("~~~~~start send_serialize end.~~~~~\n");

    printf("start client_stop_find.\n");
    ne_someip_client_context_stop_find_service(g_client_context, &g_service_spec);

    printf("start client_unsubscribe.\n");
    ne_someip_client_context_unreg_find_service_handler(g_client_context, &g_service_spec, someip_find_status_handler_callback);
    ne_someip_client_context_unreg_service_status_handler(g_client_context, &g_service_spec, someip_service_available_handler_callback);
    ne_someip_client_context_unreg_response_handler(g_client_context, g_req_instance, someip_recv_response_handler_callback);
    ne_someip_client_context_unreg_sub_handler(g_client_context, g_req_instance, someip_subscribe_status_handler_callback);
    ne_someip_client_context_unreg_send_status_handler(g_client_context, g_req_instance, someip_send_req_status_handler_callback);
    ne_someip_client_context_unreg_event_handler(g_client_context, g_req_instance, someip_recv_event_handler_callback);

    ne_someip_client_context_destroy_req_instance(g_client_context, g_req_instance);
    ne_someip_client_context_unref(g_client_context);
    ne_someip_config_release_someip_config(&g_someip_config);
}
