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
#include <stdio.h>
#include <errno.h>
#include "ne_someip_server_context.h"
#include "ne_someip_config_parse.h"
#include "NESomeIPE2EManager.h"
#include "NESomeIPE2EChecker.h"
#include "NESomeIPE2EProtector.h"

static ne_someip_server_context_t* g_server_context = NULL;
static ne_someip_provided_instance_t* g_pro_instance = NULL;
static std::shared_ptr<NESomeIPE2EChecker> e2eCheck12290 = nullptr;
static std::shared_ptr<NESomeIPE2EChecker> e2eCheck12289 = nullptr;
static std::shared_ptr<NESomeIPE2EProtector> e2eProtect12289 = nullptr;
static E2E_state_machine::E2EState g_E2EState;

ne_someip_config_t* load_config(const char* someip_config_path);
void on_service_status(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status, ne_someip_error_code_t ret_code, void* user_data);
void on_recv_req(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
void on_send_resp(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id, ne_someip_error_code_t ret_code, void* user_data);

int main(int argc, char* argv[]) {
    printf("%s: start \n", argv[0]);
    ne_someip_config_t* someip_config = load_config("./someip_server_udp.json");
    if (NULL == someip_config) {
        printf("load confog error\n");
        return -1;
    }

    g_server_context = ne_someip_server_create(false, 0, 0);
    if (NULL == g_server_context) {
        printf("create g_server_context falied\n");
        return -1;
    }

    g_pro_instance = ne_someip_server_create_instance(g_server_context, someip_config->provided_instance_array.provided_instance_config);
    if (NULL == g_pro_instance) {
        printf("create g_pro_instance falied\n");
        return -1;
    }

    ne_someip_server_reg_service_status_handler(g_server_context, g_pro_instance, on_service_status, &someip_config->provided_instance_array.provided_instance_config->service_config->service_id);
    ne_someip_server_reg_req_handler(g_server_context, g_pro_instance, on_recv_req, NULL);
    ne_someip_server_reg_resp_status_handler(g_server_context, g_pro_instance, on_send_resp, &someip_config->provided_instance_array.provided_instance_config->service_config->service_id);


    // e2e protect
    std::shared_ptr<NESomeIPE2EManager> e2eManager = NESomeIPE2EManager::getInstance();

    FILE* file_fd = fopen("./e2e_statemachines.json", "rb");
    if (nullptr == file_fd) {
        printf("open file failed\n");
        return -1;
    }

    fseek(file_fd, 0, SEEK_END);
    long data_length = ftell(file_fd);
    if (data_length <= 0) {
        printf("ftell failed errmsg:%s\n", strerror(errno));
        fclose(file_fd);
        return -1;
    }
    rewind(file_fd);

    char* file_data = reinterpret_cast<char*>(calloc(1, data_length + 1));
    if (nullptr == file_data) {
        printf("file_data calloc error\n");
        fclose(file_fd);
        return -1;
    }
    fread(file_data, 1, data_length, file_fd);
    fclose(file_fd);

    ne_someip_error_code_t ret = e2eManager->loadDataIdConfig(file_data);
    if (ne_someip_error_code_ok != ret) {
        printf("e2e loadDataIdConfig failed.\n");
        return -1;
    }
    if (nullptr != file_data) {
        free(file_data);
        file_data = nullptr;
    }

    e2eCheck12290 = e2eManager->createE2EChecker(101, ne_someip_e2e_mode_multi, true);
    if (nullptr == e2eCheck12290) {
        printf("create e2eCheck12290 failed.\n");
        return -1;
    }
    e2eCheck12289 = e2eManager->createE2EChecker(102, ne_someip_e2e_mode_multi, true);
    if (nullptr == e2eCheck12289) {
        printf("create e2eCheck12289 failed.\n");
        return -1;
    }
    e2eProtect12289 = e2eManager->createE2EProtector(102, true);
    if (nullptr == e2eProtect12289) {
        printf("create e2eProtect12289 failed.\n");
        return -1;
    }

    ret = ne_someip_server_offer(g_server_context, g_pro_instance);
    if (ne_someip_error_code_ok != ret) {
        printf("g_pro_instance offer service error.\n");
        return -1;
    }

    sleep(200);

    printf("stop offer service\n");
    ret = ne_someip_server_stop_offer(g_server_context, g_pro_instance);
    if (ne_someip_error_code_ok != ret) {
        printf("g_pro_instance stop offer service error.\n");
        return -1;
    }

    ne_someip_server_unreg_service_status_handler(g_server_context, g_pro_instance);
    ne_someip_server_unreg_req_handler(g_server_context, g_pro_instance);
    ne_someip_server_unreg_resp_status_handler(g_server_context, g_pro_instance);

    ne_someip_server_destroy_instance(g_server_context, g_pro_instance);
    ne_someip_server_unref(g_server_context);
    sleep(1);
    ne_someip_config_release_someip_config(&someip_config);
    return 0;
}

ne_someip_config_t* load_config(const char* someip_config_path) {
    FILE* file_fd = fopen(someip_config_path, "rb");
    if (NULL == file_fd) {
        printf("open file failed\n");
        return NULL;
    }

    fseek(file_fd, 0, SEEK_END);
    long data_length = ftell(file_fd);
    if (data_length <= 0) {
        printf("ftell failed errno:%d errmsg:%s\n", errno, strerror(errno));
        fclose(file_fd);
        return NULL;
    }
    rewind(file_fd);

    char* file_data = reinterpret_cast<char*>(calloc(1, data_length + 1));
    if (NULL == file_data) {
        printf("file_data calloc error\n");
        fclose(file_fd);
        return NULL;
    }
    fread(file_data, 1, data_length, file_fd);
    fclose(file_fd);

    ne_someip_config_t* someip_config = ne_someip_config_parse_someip_config_by_content(file_data);
    free(file_data);
    return someip_config;
}

void on_service_status(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status,
    ne_someip_error_code_t ret_code, void* user_data) {
    if (ne_someip_offer_status_stopped == status) {
        printf("on_service_status: service_id:[%d], ==> [STOPPED], ret:[0x%x]\n", *(ne_someip_service_id_t*)(user_data), ret_code);
    } else if (ne_someip_offer_status_pending == status) {
        printf("on_service_status: service_id:[%d], ==> [PENDING], ret:[0x%x]\n", *(ne_someip_service_id_t*)(user_data), ret_code);
    } else if (ne_someip_offer_status_running == status) {
        printf("on_service_status: service_id:[%d], ==> [RUNNING], ret:[0x%x]\n", *(ne_someip_service_id_t*)(user_data), ret_code);
    }
}

void on_recv_req(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload,
    ne_someip_remote_client_info_t* client_info, void* user_data) {
    printf("\n\non_recv_req: service_id:[%d], method_id:[%d], session_id:[%d], message_length:[%d], message_type:[0x%x]\n",
        header->service_id, header->method_id, header->session_id, header->message_length, header->message_type);
    uint32_t counter = 0;
    E2E_state_machine::E2ECheckStatus checkStatus;
    if (12290 == header->method_id) {
        checkStatus = e2eCheck12290->check(header, payload, &counter, g_E2EState);
    } else if (12289 == header->method_id) {
        checkStatus = e2eCheck12289->check(header, payload, &counter, g_E2EState);
    }
    printf("E2ECheck: counter:[%d], checkStatus:[%d], E2EState:[%d], payload:%s\n", counter, checkStatus, g_E2EState, (char*)(payload->buffer_list[0]->data));

    // send response (12289)
    if (ne_someip_message_type_request == header->message_type) {
        header->message_type = ne_someip_message_type_response;
        ne_someip_error_code_t ret = e2eProtect12289->protect(header, payload, counter);
        if (ne_someip_error_code_ok != ret) {
            printf("e2e protect error, ret:[0x%x]\n", ret);
            return;
        } else {
            printf("e2e protect success: counter:[%d], message_length:[%d]\n", counter, header->message_length);
        }
        ne_someip_server_send_response(g_server_context, instance, NULL, header, payload, client_info);
    }
}

void on_send_resp(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id,
    ne_someip_error_code_t ret_code, void* user_data) {
    printf("on_send_resp: service_id:[%d], method_id:[%d], ret:[%d]\n", *(ne_someip_service_id_t*)(user_data), method_id, ret_code);
}

/* EOF */
