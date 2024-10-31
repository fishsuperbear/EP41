/**
 * Copyright @ 2019 iAuto (Shanghai) Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * iAuto (Shanghai) Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include "someip/base/log/ne_someip_log.h"
#include "NESomeIPEtsProcess.h"
#include "NESomeIPTc8Process.h"


#define DEBUG_LOG(format, ...) \
 { \
    char print_msg[1024]= { 0 };    \
    struct timeval tv;              \
    gettimeofday(&tv, nullptr);     \
    struct tm *timeinfo = localtime(&tv.tv_sec);        \
    uint32_t milliseconds = tv.tv_usec / 1000;          \
    char time_buf[64] = { 0 };                          \
    memset(time_buf, 0x00, sizeof(time_buf));           \
    memset(print_msg, 0x00, sizeof(print_msg));         \
    snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",         \
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,             \
        timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, milliseconds);          \
    snprintf(print_msg, sizeof(print_msg), (format), ##__VA_ARGS__);                   \
    printf("[%s] [%d %ld %s@%s(%d) | %s]\n", time_buf, getpid(), syscall(__NR_gettid), \
        __FUNCTION__, (nullptr == strrchr(__FILE__, '/')) ? __FILE__: (strrchr(__FILE__, '/') + 1), __LINE__, (print_msg)); \
 }


static bool initComplete = false;
static ne_someip_instance_id_t instaceId01 = 0x0001;
static ne_someip_instance_id_t instaceId02 = 0x0002;
static ne_someip_major_version_t majorVersion1 = 1;
static ne_someip_minor_version_t minorVersion1 = 0;
static ne_someip_server_context_t* g_server_context = nullptr;
static ne_someip_config_t* g_someip_config = nullptr;
static ne_someip_provided_instance_t* g_pro_instance_2000_1 = nullptr;
static ne_someip_provided_instance_t* g_pro_instance_2000_2 = nullptr;
static ne_someip_provided_instance_t* g_pro_instance_3000_1 = nullptr;
static ne_someip_provided_instance_t* g_pro_instance_3000_2 = nullptr;
static std::shared_ptr<NESomeIPEtsProcess> g_ets = nullptr;

static ne_someip_config_t* load_config(const char* someip_config_path);
static void on_service_status(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status, ne_someip_error_code_t ret_code, void* user_data);
static void on_send_event(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id, ne_someip_error_code_t ret_code, void* user_data);
static void on_recv_subscribe(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* eventgroup_id, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_2000_1(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_2000_2(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_3000_1(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_3000_2(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_send_resp(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id, ne_someip_error_code_t ret_code, void* user_data);
static ne_someip_payload_t* create_payload();

static void* send_event_thread_func(void* payload)
{
    g_ets->triggerEventUINT8((ne_someip_payload_t*)payload);
    ne_someip_payload_unref((ne_someip_payload_t*)payload);
    return nullptr;
}

static void* suspend_thread_func(void* payload)
{
    g_ets->suspendInterface((ne_someip_payload_t*)payload);
    ne_someip_payload_unref((ne_someip_payload_t*)payload);
    return nullptr;
}



NESomeIPTc8Process::NESomeIPTc8Process() {

}

NESomeIPTc8Process::~NESomeIPTc8Process() {

}

// method id : 1, reset all interface
void resetInterface(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    if (nullptr == instance || nullptr == header) {
        ne_someip_log_error("parameters null error");
        return;
    }
    DEBUG_LOG("Recv message on serviceId [0x%x], instanceId [0x%x], methodId [0x%x]", header->service_id, 2000, header->method_id);
}

void messageHandlerETSTest(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    DEBUG_LOG("------------------------------ETS test case start------------------------------");
    if (nullptr == g_ets) {
        g_ets = std::make_shared<NESomeIPEtsProcess>(g_server_context, g_pro_instance_2000_1);
        if (nullptr == g_ets) {
            DEBUG_LOG("make_shared failed.");
            return;
        }
        g_ets->setThisPointerToEts();
    }
    // std::shared_ptr<NESomeIPEtsProcess> ets = std::make_shared<NESomeIPEtsProcess>(g_server_context, g_pro_instance_2000_1);
    ne_someip_message_type_t req_message_type = header->message_type;
    header->return_code = ne_someip_return_code_not_ok;
    ne_someip_method_id_t MethodID = header->method_id;
    DEBUG_LOG("MethodID is %d", MethodID);
    switch (MethodID) {
    case 0x1F:
        header->return_code = g_ets->checkByteOrder(payload);
        break;
    case 0x2F:
        header->return_code = g_ets->clientServiceActivate(payload);
        break;
    case 0x30:
        header->return_code = g_ets->clientServiceDeactivate(payload);
        return;
    case 0x32:
        header->return_code = g_ets->clientServiceSubscribeEventgroup(payload);
        break;
    case 0x23:
        header->return_code = g_ets->echoCommonDatatypes(payload);
        break;
    case 0x17:
        header->return_code = g_ets->echoENUM(payload);
        break;
    case 0x12:
        header->return_code = g_ets->echoFLOAT64(payload);
        break;
    case 0x34:
        header->return_code = g_ets->echoInt64(payload);
        break;
    case 0x0e:
        header->return_code = g_ets->echoINT8(payload);
        break;
    case 0x36:
        header->return_code = g_ets->echoStaticUINT8Array(payload);
        break;
    case 0x08:
        header->return_code = g_ets->echoUINT8(payload);
        break;
    case 0x09:
        header->return_code = g_ets->echoUINT8Array(payload);
        break;
    case 0x3E:
        header->return_code = g_ets->echoUINT8Array8BitLength(payload);
        break;
    case 0x3F:
        header->return_code = g_ets->echoUINT8Array16BitLength(payload);
        break;
    case 0x35:
        {
            header->return_code = g_ets->echoUINT8Array2Dim(payload);
            if (header->return_code != ne_someip_return_code_ok) {
                header->message_type = ne_someip_message_type_error;
            }
        }
        break;
    case 0x37:
        {
            header->return_code = g_ets->echoUINT8ArrayMinSize(payload);
            if (header->return_code != ne_someip_return_code_ok) {
                header->message_type = ne_someip_message_type_error;
            }
        }
        break;
    case 0x0A:
        header->return_code = g_ets->echoUINT8(payload); // test case echoUINT8RELIABLE
        break;
    case 0x19:
        header->return_code = g_ets->echoUNION(payload);
        break;
    case 0x16:
        {
            header->return_code = g_ets->echoUTF16DYNAMIC(payload);
            if (header->return_code != ne_someip_return_code_ok) {
                header->message_type = ne_someip_message_type_error;
            }
        }
        break;
    case 0x14:
        {
            header->return_code = g_ets->echoUTF16FIXED(payload);
            if (header->return_code != ne_someip_return_code_ok) {
                header->message_type = ne_someip_message_type_error;
            }
        }
        break;
    case 0x15:
        {
            header->return_code = g_ets->echoUTF8DYNAMIC(payload);
            if (header->return_code != ne_someip_return_code_ok) {
                header->message_type = ne_someip_message_type_error;
            }
        }
        break;
    case 0x13:
        header->return_code = g_ets->echoUTF8FIXED(payload);
        break;
    case 0x02:
        {
            pthread_t th_test;
            int* thread_ret = nullptr;
            ne_someip_payload_ref(payload);
            pthread_create(&th_test, NULL, suspend_thread_func, payload);
            pthread_detach(th_test);
            // header->return_code = g_ets->suspendInterface(payload);
            break;
        }
    case 0x03:
        {
            pthread_t th_test;
            int* thread_ret = nullptr;
            ne_someip_payload_ref(payload);
            pthread_create(&th_test, NULL, send_event_thread_func, payload);
            pthread_detach(th_test);
            // pthread_join(th_test, (void**)&thread_ret);
            // header->return_code = g_ets->triggerEventUINT8(payload);
            break;
        }
    case 0x04:
        header->return_code = g_ets->triggerEventUINT8Array(payload);
        break;
    case 0x05:
        header->return_code = g_ets->triggerEventUINT8Reliable(payload);
        break;
    case 0x3a:
        header->return_code = g_ets->triggerEventUINT8Multicast(payload);
        break;
    case 0x41 :
        header->return_code = g_ets->echoBitfields(payload);
        break;
    case 0x3B:
        header->return_code = g_ets->clientServiceGetLastValueEventTCP(&payload);
        break;
    case 0x3C:
        header->return_code = g_ets->clientServiceGetLastValueEventUDPUnicast(&payload);
        break;
    case 0x3D:
        header->return_code = g_ets->clientServiceGetLastValueEventUDPMulticast(&payload);
        break;
    case 0x25:
        header->return_code = g_ets->interfaceVersionGetter(&header, &payload);
        break;
    case 0x26:
        header->return_code = g_ets->testFieldUint8Getter(&header, &payload);
        break;
    case 0x27:
        header->return_code = g_ets->testFieldUint8Setter(&header, &payload);
        break;
    case 0x28:
        header->return_code = g_ets->testFieldUint8ArrayGetter(&header, &payload);
        break;
    case 0x29:
        header->return_code = g_ets->testFieldUint8ArraySetter(&header, &payload);
        break;
    case 0x2A:
        header->return_code = g_ets->testFieldUint8ReliableGetter(&header, &payload);
        break;
    case 0x2B:
        header->return_code = g_ets->testFieldUint8ReliableSetter(&header, &payload);
        break;
    case 0x8001:
        header->return_code = g_ets->TestEventUINT8(payload);
        break;
    case 0x8003:
        header->return_code = g_ets->TestEventUINT8Reliable(payload);
        break;
    case 0x800B:
        header->return_code = g_ets->TestEventUINT8Multicast(payload);
        break;
    case 0x1:
        g_ets->testResetInterface();
        return;
        break;
    }

    if (ne_someip_message_type_request_no_return != req_message_type) {
        header->message_type = (ne_someip_return_code_ok == header->return_code ? ne_someip_message_type_response : ne_someip_message_type_error);
        if(ne_someip_return_code_ok != header->return_code /* && header->method_id != 0x16 */) {
            ne_someip_payload_t *payload1 = ne_someip_payload_create();
            payload1->buffer_list = (ne_someip_payload_slice_t **)malloc(sizeof(ne_someip_payload_slice_t));
            ne_someip_payload_slice_t *payload_slice = (ne_someip_payload_slice_t *)malloc(sizeof(ne_someip_payload_slice_t));
            uint8_t tem_buffer[1] = "";
            uint8_t *data = (uint8_t *)malloc(1 * sizeof(uint8_t));
            memset(data, 0, 1);
            memcpy(data, tem_buffer, 1);
            payload_slice->free_pointer = data;
            payload_slice->data = data;
            payload_slice->length = 0;

            payload1->buffer_list[0] = payload_slice;
            payload1->num = 1;
            payload = payload1;
        }

        if (ne_someip_error_code_ok != ne_someip_server_send_response(g_server_context, instance, nullptr, header, payload, client_info)) {
            DEBUG_LOG("ne_someip_server_send_response failure");
        } else {
            DEBUG_LOG("ne_someip_server_send_response success");
        }
    }

    DEBUG_LOG("--------------------------------ETS test case end------------------------------");
}

bool NESomeIPTc8Process::init() {
    DEBUG_LOG("init start");
    // create server context
    g_server_context = ne_someip_server_create(false, 0, 0);
    if (nullptr == g_server_context) {
        DEBUG_LOG("create g_server_context falied");
        return false;
    }

    // load config
    g_someip_config = load_config("conf/someip_config_ets.json");
    if (nullptr == g_someip_config) {
        DEBUG_LOG("load someip config error");
        return false;
    }

    // create pro_instance
    g_pro_instance_2000_1 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 0);
    g_pro_instance_2000_2 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 1);
    g_pro_instance_3000_1 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 2);
    g_pro_instance_3000_2 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 3);
    if (nullptr == g_pro_instance_2000_1 || nullptr == g_pro_instance_2000_2 || nullptr == g_pro_instance_3000_1 || nullptr == g_pro_instance_3000_2) {
        DEBUG_LOG("create g_pro_instance falied");
        return false;
    }

    DEBUG_LOG("server: register messsage handler start");
    ne_someip_provided_instance_t* pro_instance[4] = {g_pro_instance_2000_1, g_pro_instance_2000_2, g_pro_instance_3000_1, g_pro_instance_3000_2};
    for (uint8_t i = 0; i < 4; i++) {
        ne_someip_server_reg_service_status_handler(g_server_context, pro_instance[i], on_service_status, &(g_someip_config->provided_instance_array.provided_instance_config + i)->service_config->service_id);
        ne_someip_server_reg_event_status_handler(g_server_context, pro_instance[i], on_send_event, &(g_someip_config->provided_instance_array.provided_instance_config + i)->service_config->service_id);
        ne_someip_server_reg_subscribe_handler(g_server_context, pro_instance[i], on_recv_subscribe, &(g_someip_config->provided_instance_array.provided_instance_config + i)->service_config->service_id);
        ne_someip_server_reg_resp_status_handler(g_server_context, pro_instance[i], on_send_resp, &(g_someip_config->provided_instance_array.provided_instance_config + i)->service_config->service_id);
    }
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[0], on_recv_req_2000_1, nullptr);
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[1], on_recv_req_2000_2, nullptr);
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[2], on_recv_req_3000_1, nullptr);
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[3], on_recv_req_3000_2, nullptr);

    // ETS start
    std::vector<uint16_t>::iterator it = MethodIDType.begin();
    for (; it != MethodIDType.end(); ++it) {
        switch (*it) {
        // case 0x01 :
        // {
        //     // 312, 1
        //     if (ne_someip_error_code_ok != ne_someip_server_reg_req_handler(g_server_context, pro_instance[0], resetInterface, nullptr)) {
        //         DEBUG_LOG("register resetInterface failure");
        //     } else {
        //         DEBUG_LOG("register resetInterface success");
        //     }
        //     break;
        // }
        case 0x8001:
        case 0x8003:
        case 0x800B:
        {
            // 312, 224
            if (ne_someip_error_code_ok != ne_someip_server_reg_req_handler(g_server_context, pro_instance[0], messageHandlerETSTest, nullptr)) {
                DEBUG_LOG("register resetInterface failure");
            } else {
                DEBUG_LOG("register resetInterface success");
            }
            break;
        }
        default :
        {
            // 312, 1
            if (ne_someip_error_code_ok != ne_someip_server_reg_req_handler(g_server_context, pro_instance[0], messageHandlerETSTest, nullptr)) {
                DEBUG_LOG("register 0x%x failure", *it);
            } else {
                DEBUG_LOG("register 0x%x success", *it);
            }
        }
        }
    }
    // ETS end

    DEBUG_LOG("server: register messsage handler end");

    initComplete = true;
    DEBUG_LOG("init success");

    return true;
}

int NESomeIPTc8Process::OfferService(uint16_t service, uint16_t NumInstance) {
    DEBUG_LOG("OfferService start, serviceId [%d], NumInstance [%d]", service, NumInstance);
    if (!initComplete) {
        return -1;
    }

    if (nullptr == g_server_context || nullptr == g_pro_instance_2000_1) {
        ne_someip_log_error("g_server_context or g_pro_instance_2000_1 nullptr");
        return -1;
    }

    if (1 == NumInstance) {
        if (service == (g_someip_config->provided_instance_array.provided_instance_config + 0)->service_config->service_id) {
            ne_someip_server_offer(g_server_context, g_pro_instance_2000_1);
        } else {
            ne_someip_server_offer(g_server_context, g_pro_instance_3000_1);
        }
        return 0;
    } else if (2 == NumInstance) {
        if (service == (g_someip_config->provided_instance_array.provided_instance_config + 0)->service_config->service_id) {
            ne_someip_server_offer(g_server_context, g_pro_instance_2000_1);
            ne_someip_server_offer(g_server_context, g_pro_instance_2000_2);
        } else {
            ne_someip_server_offer(g_server_context, g_pro_instance_3000_1);
            ne_someip_server_offer(g_server_context, g_pro_instance_3000_2);
        }
        return 0;
    } else {
        ne_someip_log_error("NumInstance [%d] error", NumInstance);
        return -1;
    }
}

int NESomeIPTc8Process::StopService(uint16_t service) {
    DEBUG_LOG("StopService start, serviceId [%d]", service);
    if (!initComplete) {
        return -1;
    }

    if (nullptr == g_server_context || nullptr == g_pro_instance_2000_1) {
        ne_someip_log_error("g_server_context or g_pro_instance_2000_1 nullptr");
        return -1;
    }

    if (service == (g_someip_config->provided_instance_array.provided_instance_config + 0)->service_config->service_id) {
        if (nullptr == g_pro_instance_2000_1 || nullptr == g_pro_instance_2000_2) {
            ne_someip_log_error("g_pro_instance nullptr");
            return -1;
        }
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_2000_1);
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_2000_2);
    } else {
        if (nullptr == g_pro_instance_3000_1 || nullptr == g_pro_instance_3000_2) {
            ne_someip_log_error("g_pro_instance nullptr");
            return -1;
        }
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_3000_1);
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_3000_2);
    }

    return 0;
}

int NESomeIPTc8Process::TriggerEvent(uint16_t service, uint16_t EventGroup, uint16_t EventId) {
    DEBUG_LOG("TriggerEvent start, serviceId [%d], EventGroup [%d], EventId [%d]", service, EventGroup, EventId);

    if (!initComplete) {
        ne_someip_log_error("init is not completed");
        return -1;
    }

    ne_someip_payload_t* payload1 =  ne_someip_payload_create();
    ne_someip_payload_slice_t* slice1 = (ne_someip_payload_slice_t*)calloc(1, sizeof(ne_someip_payload_slice_t));
    if (nullptr == slice1) {
        DEBUG_LOG("slice calloc error");
        return -1;
    }
    uint32_t payload_len = 10;
    uint8_t* data1 = (uint8_t*)calloc(1, payload_len);
    uint8_t aa[] = "zhongguo1";
    memcpy(data1, aa, payload_len);
    slice1->data = data1;
    slice1->length = payload_len;
    slice1->free_pointer = data1;
    payload1->buffer_list = (ne_someip_payload_slice_t**)calloc(1, sizeof(ne_someip_payload_slice_t*));
    *(payload1->buffer_list) = slice1;
    payload1->num = 1;
    ne_someip_header_t header1;
    if (service == (g_someip_config->provided_instance_array.provided_instance_config + 0)->service_config->service_id) {
        if (1 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_2000_1, EventId, &header1);
        }
        if (3 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_2000_2, EventId, &header1);
        }
    } else {
        if (5 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_3000_1, EventId, &header1);
        }
        if (6 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_3000_2, EventId, &header1);
        }
    }
    ne_someip_error_code_t ret = ne_someip_server_send_event(g_server_context, g_pro_instance_2000_1, nullptr, &header1, payload1);
    ne_someip_payload_unref(payload1);

    if (ne_someip_error_code_ok != ret) {
        DEBUG_LOG("send event error, serviceId [%d], EventGroup [%d], EventId [%d]", service, EventGroup, EventId);
        return -1;
    } else {
        DEBUG_LOG("send event success, serviceId [%d], EventGroup [%d], EventId [%d]", service, EventGroup, EventId);
    }
    return 0;
}

ne_someip_config_t* load_config(const char* someip_config_path)
{
    FILE* file_fd = fopen(someip_config_path, "rb");
    if (nullptr == file_fd) {
        DEBUG_LOG("open file failed");
        return nullptr;
    }

    fseek(file_fd, 0, SEEK_END);
    long data_length = ftell(file_fd);
    if (data_length <= 0) {
        DEBUG_LOG("ftell failed errno:%d errmsg:%s", errno, strerror(errno));
        fclose(file_fd);
        return nullptr;
    }
    rewind(file_fd);

    char* file_data = (char*)calloc(1, data_length + 1);
    if (nullptr == file_data) {
        DEBUG_LOG("file_data calloc error");
        fclose(file_fd);
        return nullptr;
    }
    fread(file_data, 1, data_length, file_fd);
    fclose(file_fd);

    ne_someip_config_t* someip_config = ne_someip_config_parse_someip_config_by_content(file_data);
    free(file_data);
    return someip_config;
}

void on_service_status(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status, ne_someip_error_code_t ret_code, void* user_data) {
    if (ne_someip_offer_status_stopped == status) {
        DEBUG_LOG("on_service_status: service_id:[%d], ===> [STOPPED], ret:[0x%x]", *(ne_someip_service_id_t*)(user_data), ret_code);
    } else if (ne_someip_offer_status_pending == status) {
        DEBUG_LOG("on_service_status: service_id:[%d], ===> [PENDING], ret:[0x%x]", *(ne_someip_service_id_t*)(user_data), ret_code);
    } else if (ne_someip_offer_status_running == status) {
        DEBUG_LOG("on_service_status: service_id:[%d], ===> [RUNNING], ret:[0x%x]", *(ne_someip_service_id_t*)(user_data), ret_code);
    }
}

void on_send_event(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id,
    ne_someip_error_code_t ret_code, void* user_data) {
    DEBUG_LOG("on_send_event: service_id:[%d], event_id:[%d], ret:[0x%x]", *(ne_someip_service_id_t*)(user_data), event_id - 0x8000, ret_code);
}

void on_recv_subscribe(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* eventgroup_id,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    DEBUG_LOG("on_recv_subscribe: service_id:[%d], remote_client:[%d:%d]",
        *(ne_someip_service_id_t*)(user_data), client_info->ipv4, client_info->port);
}

void on_recv_req_2000_1(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    if (2000 != header->service_id) {
        DEBUG_LOG("on_recv_req_2000_1: service_id:[%d] error", header->service_id);
        return;
    }
    DEBUG_LOG("on_recv_req_2000_1: service_id:[%d], method_id:[%d], session_id:[%d], message_length:[%d], message_type:[0x%x]",
        header->service_id, header->method_id, header->session_id, header->message_length, header->message_type);

    ne_someip_payload_t* payload1 = create_payload();

    if (1 == header->method_id) {
        // fire and forget
        // todo nothing
    } else if (21 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, nullptr, header, payload1, client_info);
    } else if (24 == header->method_id) {
        // setter
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, nullptr, header, payload1, client_info);
    } else if (23 == header->method_id) {
        // getter
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, nullptr, header, payload1, client_info);
    }

    ne_someip_payload_unref(payload1);
}

void on_recv_req_2000_2(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    if (2000 != header->service_id) {
        DEBUG_LOG("on_recv_req_2000_2: service_id:[%d] error", header->service_id);
        return;
    }
    DEBUG_LOG("on_recv_req_2000_2: service_id:[%d], method_id:[%d], session_id:[%d], message_length:[%d], message_type:[0x%x]",
        header->service_id, header->method_id, header->session_id, header->message_length, header->message_type);

    ne_someip_payload_t* payload1 = create_payload();
    if (21 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, nullptr, header, payload1, client_info);
    }

    ne_someip_payload_unref(payload1);
}

void on_recv_req_3000_1(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    if (3000 != header->service_id) {
        DEBUG_LOG("on_recv_req_3000_1: service_id:[%d] error", header->service_id);
        return;
    }
    DEBUG_LOG("on_recv_req_3000_1: service_id:[%d], method_id:[%d], session_id:[%d], message_length:[%d], message_type:[0x%x]",
        header->service_id, header->method_id, header->session_id, header->message_length, header->message_type);

    ne_someip_payload_t* payload1 = create_payload();
    if (31 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, nullptr, header, payload1, client_info);
    }

    ne_someip_payload_unref(payload1);
}

void on_recv_req_3000_2(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    if (3000 != header->service_id) {
        DEBUG_LOG("on_recv_req_3000_2: service_id:[%d] error", header->service_id);
        return;
    }
    DEBUG_LOG("on_recv_req_3000_2: service_id:[%d], method_id:[%d], session_id:[%d], message_length:[%d], message_type:[0x%x]",
        header->service_id, header->method_id, header->session_id, header->message_length, header->message_type);

    ne_someip_payload_t* payload1 = create_payload();
    if (31 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, nullptr, header, payload1, client_info);
    }

    ne_someip_payload_unref(payload1);
}

void on_send_resp(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id,
    ne_someip_error_code_t ret_code, void* user_data) {
    DEBUG_LOG("on_send_resp: service_id:[%d], method_id:[%d], ret:[0x%x]", *(ne_someip_service_id_t*)(user_data), method_id, ret_code);
}

ne_someip_payload_t* create_payload() {
    ne_someip_payload_t* payload1 =  ne_someip_payload_create();
    ne_someip_payload_slice_t* slice1 = (ne_someip_payload_slice_t*)calloc(1, sizeof(ne_someip_payload_slice_t));
    if (nullptr == slice1) {
        DEBUG_LOG("slice calloc error");
        return nullptr;
    }
    uint32_t payload_len = 4;
    uint8_t* data1 = (uint8_t*)calloc(1, payload_len);
    uint8_t aa[4] = {0};
    memcpy(data1, aa, payload_len);
    slice1->data = data1;
    slice1->length = payload_len;
    slice1->free_pointer = data1;
    payload1->buffer_list = (ne_someip_payload_slice_t**)calloc(1, sizeof(ne_someip_payload_slice_t*));
    if (nullptr == payload1->buffer_list) {
        DEBUG_LOG("payload1->buffer_list calloc error ");
        free(payload1);
        return nullptr;
    }
    *(payload1->buffer_list) = slice1;
    payload1->num = 1;

    return payload1;
}

/* EOF */
