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
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include "someip/base/log/ne_someip_log.h"
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
static ne_someip_server_context_t* g_server_context = NULL;
static ne_someip_config_t* g_someip_config = NULL;
static ne_someip_config_t* g_someip_config2 = NULL;
static ne_someip_provided_instance_t* g_pro_instance_2000_1 = NULL;
static ne_someip_provided_instance_t* g_pro_instance_2000_2 = NULL;
static ne_someip_provided_instance_t* g_pro_instance_3000_1 = NULL;
static ne_someip_provided_instance_t* g_pro_instance_3000_2 = NULL;

static ne_someip_config_t* load_config(const char* someip_config_path);
static void on_service_status(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status, ne_someip_error_code_t ret_code, void* user_data);
static void on_send_event(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id, ne_someip_error_code_t ret_code, void* user_data);
static void on_recv_subscribe(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* eventgroup_id, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_2000_1(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_2000_2(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_3000_1(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_recv_req_3000_2(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
static void on_send_resp(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id, ne_someip_error_code_t ret_code, void* user_data);
static ne_someip_payload_t* create_payload(ne_someip_method_id_t method_id);

bool init() {
    DEBUG_LOG("init start");
    // create server context
    g_server_context = ne_someip_server_create(false, 0, 0);
    if (NULL == g_server_context) {
        DEBUG_LOG("create g_server_context falied");
        return false;
    }

    // load config
    g_someip_config = load_config("conf/someip_config_tc8.json");
    if (NULL == g_someip_config) {
        DEBUG_LOG("load someip config error");
        return false;
    }

    // create pro_instance
    g_pro_instance_2000_1 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 0);
    g_pro_instance_2000_2 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 1);
    g_pro_instance_3000_1 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 2);
    g_pro_instance_3000_2 = ne_someip_server_create_instance(g_server_context, g_someip_config->provided_instance_array.provided_instance_config + 3);
    if (NULL == g_pro_instance_2000_1 || NULL == g_pro_instance_2000_2 || NULL == g_pro_instance_3000_1 || NULL == g_pro_instance_3000_2) {
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
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[0], on_recv_req_2000_1, NULL);
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[1], on_recv_req_2000_2, NULL);
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[2], on_recv_req_3000_1, NULL);
    ne_someip_server_reg_req_handler(g_server_context, pro_instance[3], on_recv_req_3000_2, NULL);
    DEBUG_LOG("server: register messsage handler end");

    initComplete = true;
    DEBUG_LOG("init success");

    return true;
}

void stopSeviceIns2000()
{
    DEBUG_LOG("stopSeviceIns2000 begin");
    ne_someip_server_unreg_req_handler(g_server_context, g_pro_instance_2000_1);
    ne_someip_server_unreg_service_status_handler(g_server_context, g_pro_instance_2000_1);
    ne_someip_server_unreg_event_status_handler(g_server_context, g_pro_instance_2000_1);
    ne_someip_server_unreg_subscribe_handler(g_server_context, g_pro_instance_2000_1);
    ne_someip_server_unreg_resp_status_handler(g_server_context, g_pro_instance_2000_1);
    ne_someip_server_destroy_instance(g_server_context, g_pro_instance_2000_1);
    g_pro_instance_2000_1 = NULL;
    DEBUG_LOG("stopSeviceIns2000 end");
}

void startSeviceIns2000(char* config_path)
{
    // reload config
    DEBUG_LOG("startSeviceIns2000 begin");
    g_someip_config2 = load_config(config_path);
    if (NULL == g_someip_config2) {
        DEBUG_LOG("reload someip config error");
        return;
    }

    g_pro_instance_2000_1 = ne_someip_server_create_instance(g_server_context, g_someip_config2->provided_instance_array.provided_instance_config + 0);
    DEBUG_LOG("server: register messsage handler start");
    ne_someip_server_reg_service_status_handler(g_server_context, g_pro_instance_2000_1, on_service_status, &(g_someip_config2->provided_instance_array.provided_instance_config + 0)->service_config->service_id);
    ne_someip_server_reg_event_status_handler(g_server_context, g_pro_instance_2000_1, on_send_event, &(g_someip_config2->provided_instance_array.provided_instance_config + 0)->service_config->service_id);
    ne_someip_server_reg_subscribe_handler(g_server_context, g_pro_instance_2000_1, on_recv_subscribe, &(g_someip_config2->provided_instance_array.provided_instance_config + 0)->service_config->service_id);
    ne_someip_server_reg_resp_status_handler(g_server_context, g_pro_instance_2000_1, on_send_resp, &(g_someip_config2->provided_instance_array.provided_instance_config + 0)->service_config->service_id);
    ne_someip_server_reg_req_handler(g_server_context, g_pro_instance_2000_1, on_recv_req_2000_1, NULL);
    DEBUG_LOG("server: register messsage handler end");
    DEBUG_LOG("startSeviceIns2000 end");
}

int OfferService(uint16_t service, uint16_t NumInstance) {
    DEBUG_LOG("OfferService start, serviceId [%d], NumInstance [%d]", service, NumInstance);
    if (!initComplete) {
        return -1;
    }

    if (NULL == g_server_context || NULL == g_pro_instance_2000_1) {
        ne_someip_log_error("g_server_context or g_pro_instance_2000_1 NULL");
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

int StopService(uint16_t service) {
    DEBUG_LOG("StopService start, serviceId [%d]", service);
    if (!initComplete) {
        return -1;
    }

    if (NULL == g_server_context || NULL == g_pro_instance_2000_1) {
        ne_someip_log_error("g_server_context or g_pro_instance_2000_1 NULL");
        return -1;
    }

    if (service == (g_someip_config->provided_instance_array.provided_instance_config + 0)->service_config->service_id) {
        if (NULL == g_pro_instance_2000_1 || NULL == g_pro_instance_2000_2) {
            ne_someip_log_error("g_pro_instance NULL");
            return -1;
        }
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_2000_1);
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_2000_2);
    } else {
        if (NULL == g_pro_instance_3000_1 || NULL == g_pro_instance_3000_2) {
            ne_someip_log_error("g_pro_instance NULL");
            return -1;
        }
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_3000_1);
        ne_someip_server_stop_offer(g_server_context, g_pro_instance_3000_2);
    }

    return 0;
}

int TriggerEvent(uint16_t service, uint16_t EventGroup, uint16_t EventId) {
    DEBUG_LOG("TriggerEvent start, serviceId [%d], EventGroup [%d], EventId [%d]", service, EventGroup, EventId);

    if (!initComplete) {
        ne_someip_log_error("init is not completed");
        return -1;
    }

    ne_someip_payload_t* payload1 =  ne_someip_payload_create();
    ne_someip_payload_slice_t* slice1 = (ne_someip_payload_slice_t*)calloc(1, sizeof(ne_someip_payload_slice_t));
    if (NULL == slice1) {
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
        if (2 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_2000_1, EventId & 0x7FFF, &header1);
        }
        if (3 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_2000_2, EventId & 0x7FFF, &header1);
        }
    } else {
        if (5 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_3000_1, EventId & 0x7FFF, &header1);
        }
        if (6 == EventGroup) {
            ne_someip_server_create_notify_header(g_pro_instance_3000_2, EventId & 0x7FFF, &header1);
        }
    }

    ne_someip_error_code_t ret = ne_someip_error_code_failed;
    if (2 == EventGroup && 0x8005 == EventId) {
        int counter = 1;
        while (counter <= 30) {
            ne_someip_server_send_event(g_server_context, g_pro_instance_2000_1, NULL, &header1, payload1);
            usleep(300*1000);
            counter++;
        }
    }

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
    DEBUG_LOG("load config start");
    FILE* file_fd = fopen(someip_config_path, "rb");
    if (NULL == file_fd) {
        DEBUG_LOG("open file failed");
        return NULL;
    }

    fseek(file_fd, 0, SEEK_END);
    long data_length = ftell(file_fd);
    if (data_length <= 0) {
        DEBUG_LOG("ftell failed errno:%d errmsg:%s", errno, strerror(errno));
        fclose(file_fd);
        return NULL;
    }
    rewind(file_fd);

    char* file_data = (char*)calloc(1, data_length + 1);
    if (NULL == file_data) {
        DEBUG_LOG("file_data calloc error");
        fclose(file_fd);
        return NULL;
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

    ne_someip_payload_t* payload1 = create_payload(header->method_id);

    if (1 == header->method_id) {
        // fire and forget
        // todo nothing
    } else if (21 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, NULL, header, payload, client_info);
    } else if (24 == header->method_id) {
        // setter
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, NULL, header, payload, client_info);
    } else if (23 == header->method_id) {
        // getter
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, NULL, header, payload, client_info);
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

    ne_someip_payload_t* payload1 = create_payload(header->method_id);
    if (21 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, NULL, header, payload, client_info);
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

    ne_someip_payload_t* payload1 = create_payload(header->method_id);
    if (31 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, NULL, header, payload, client_info);
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

    ne_someip_payload_t* payload1 = create_payload(header->method_id);
    if (31 == header->method_id) {
        // request and response
        header->message_type = ne_someip_message_type_response;
        header->return_code = 0x00;
        ne_someip_server_send_response(g_server_context, instance, NULL, header, payload, client_info);
    }

    ne_someip_payload_unref(payload1);
}

void on_send_resp(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id,
    ne_someip_error_code_t ret_code, void* user_data) {
    DEBUG_LOG("on_send_resp: service_id:[%d], method_id:[%d], ret:[0x%x]", *(ne_someip_service_id_t*)(user_data), method_id, ret_code);
}

ne_someip_payload_t* create_payload(ne_someip_method_id_t method_id) {
    ne_someip_payload_t* payload1 =  ne_someip_payload_create();
    ne_someip_payload_slice_t* slice1 = (ne_someip_payload_slice_t*)calloc(1, sizeof(ne_someip_payload_slice_t));
    if (NULL == slice1) {
        DEBUG_LOG("slice calloc error");
        return NULL;
    }

    uint32_t payload_len = 8;
    uint8_t* data1 = (uint8_t*)calloc(1, payload_len);
    uint8_t aa[8] = {0};
    memcpy(data1, aa, payload_len);
    slice1->data = data1;
    slice1->length = payload_len;
    slice1->free_pointer = data1;
    payload1->buffer_list = (ne_someip_payload_slice_t**)calloc(1, sizeof(ne_someip_payload_slice_t*));
    if (NULL == payload1->buffer_list) {
        DEBUG_LOG("payload1->buffer_list calloc error ");
        free(payload1);
        return NULL;
    }
    *(payload1->buffer_list) = slice1;
    payload1->num = 1;

    return payload1;
}

/* EOF */
