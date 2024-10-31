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
#include "ne_someip_config_parse.h"
#include "ne_someip_client_context.h"
#include "NESomeIPE2EManager.h"
#include "NESomeIPE2EProtector.h"
#include "NESomeIPE2EChecker.h"

static ne_someip_client_context_t* g_client_context = nullptr;
static ne_someip_required_service_instance_t* g_req_instance = nullptr;
static std::shared_ptr<NESomeIPE2EChecker> e2eCheck12289 = nullptr;
static uint32_t g_send_counter_12289 = 0;
static E2E_state_machine::E2EState g_E2EState;

ne_someip_config_t* load_config(const char* someip_config_path);
ne_someip_required_service_instance_t*
create_client_req_instance(const ne_someip_required_service_instance_config_t* inst_config);
void on_find_status(const ne_someip_find_offer_service_spec_t* spec, ne_someip_find_status_t status,
    ne_someip_error_code_t code, void* user_data);
void reg_find_service_handler(const ne_someip_required_service_instance_config_t* inst_config);
void unreg_find_service_handler(const ne_someip_required_service_instance_config_t* inst_config);
void on_service_available(const ne_someip_find_offer_service_spec_t* spec, ne_someip_service_status_t status,
    void* user_data);
void reg_service_status_handler(const ne_someip_required_service_instance_config_t* inst_config);
void unreg_service_status_handler(const ne_someip_required_service_instance_config_t* inst_config);
ne_someip_error_code_t start_find_service(const ne_someip_required_service_instance_config_t* inst_config);
ne_someip_error_code_t stop_find_service(const ne_someip_required_service_instance_config_t* inst_config);
void on_recv_response(ne_someip_required_service_instance_t* instance, ne_someip_header_t* header,
    ne_someip_payload_t* payload, void* user_data);
void on_send_req(ne_someip_required_service_instance_t* instance, void* seq_def, ne_someip_method_id_t method_id,
    ne_someip_error_code_t code, void* user_data);
ne_someip_error_code_t create_and_send_req(const std::shared_ptr<NESomeIPE2EProtector>& e2eProtect);

int main(int argc, char* argv[]) {
    ne_someip_config_t* someip_config = load_config("./someip_client_udp.json");
    if (NULL == someip_config) {
        printf("load config error\n");
        return -1;
    }

    g_client_context = ne_someip_client_context_create(0, true, 0, 0);
    if (NULL == g_client_context) {
        printf("create g_client_context error.\n");
        return -1;
    }

    g_req_instance = create_client_req_instance(someip_config->required_instance_array.required_instance_config);
    if (NULL == g_req_instance) {
        printf("create g_req_instance falied\n");
        return -1;
    }

    reg_find_service_handler(someip_config->required_instance_array.required_instance_config);
    reg_service_status_handler(someip_config->required_instance_array.required_instance_config);
    ne_someip_client_context_reg_send_status_handler(g_client_context, g_req_instance, on_send_req, NULL);
    ne_someip_client_context_reg_response_handler(g_client_context, g_req_instance, on_recv_response, NULL);


    // e2e protect
    std::shared_ptr<NESomeIPE2EManager> e2eManager = NESomeIPE2EManager::getInstance();

    FILE* file_fd = fopen("./e2e_statemachines.json", "rb");
    if (nullptr == file_fd) {
        printf("open file failed\n");
        return -1;
    }

    fseek(file_fd, 0, SEEK_END);
    int64_t data_length = ftell(file_fd);
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

    std::shared_ptr<NESomeIPE2EProtector> e2eProtect12290 = e2eManager->createE2EProtector(101, false);
    if (nullptr == e2eProtect12290) {
        printf("create e2eProtect12290 failed.\n");
        return -1;
    }
    std::shared_ptr<NESomeIPE2EProtector> e2eProtect12289 = e2eManager->createE2EProtector(102, false);
    if (nullptr == e2eProtect12289) {
        printf("create e2eProtect12289 failed.\n");
        return -1;
    }
    e2eCheck12289 = e2eManager->createE2EChecker(102, ne_someip_e2e_mode_single, false);
    if (nullptr == e2eCheck12289) {
        printf("create e2eCheck12289 failed.\n");
        return -1;
    }

    {
        uint32_t curCounter = 0;
        E2E_state_machine::E2ECheckStatus checkStatus = e2eCheck12289->check(nullptr, nullptr, &curCounter, g_E2EState);
        printf("E2ECheck end: curCounter:[%d], g_send_counter_12289:[%d], checkStatus:[%d], E2EState:[%d]\n",
            curCounter, g_send_counter_12289, checkStatus, g_E2EState);
    }

    // find service
    ret = start_find_service(someip_config->required_instance_array.required_instance_config);
    if (ne_someip_error_code_ok != ret) {
        printf("find service failed.\n");
        return -1;
    }

    sleep(2);

    for (size_t i = 0; i < 10; i++) {
        // create_and_send_req(e2eProtect12290);
        create_and_send_req(e2eProtect12289);
        sleep(5);
    }

    sleep(10);

    ret = stop_find_service(someip_config->required_instance_array.required_instance_config);
    if (ne_someip_error_code_ok != ret) {
        printf("g_req_instance stop find service failed.\n");
        return -1;
    }

    unreg_find_service_handler(someip_config->required_instance_array.required_instance_config);
    unreg_service_status_handler(someip_config->required_instance_array.required_instance_config);
    ne_someip_client_context_unreg_response_handler(g_client_context, g_req_instance, on_recv_response);
    ne_someip_client_context_unreg_send_status_handler(g_client_context, g_req_instance, on_send_req);

    ne_someip_client_context_destroy_req_instance(g_client_context, g_req_instance);
    ne_someip_client_context_unref(g_client_context);
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
    int64_t data_length = ftell(file_fd);
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

ne_someip_required_service_instance_t*
create_client_req_instance(const ne_someip_required_service_instance_config_t* inst_config) {
    if (NULL == inst_config) {
        printf("inst_config null, error\n");
        return NULL;
    }

    ne_someip_service_instance_spec_t spec;
    spec.service_id = inst_config->service_config->service_id;
    spec.instance_id = inst_config->instance_id;
    spec.major_version = inst_config->service_config->major_version;
    return ne_someip_client_context_create_req_instance(g_client_context, inst_config, &spec);
}

void on_find_status(const ne_someip_find_offer_service_spec_t* spec, ne_someip_find_status_t status,
    ne_someip_error_code_t code, void* user_data) {
    if (ne_someip_find_status_stopped == status) {
        printf("on_find_status: [0x%x:%d:%d:%d] ==> [STOPPED], ret:[%d]\n", spec->ins_spec.service_id,
            spec->ins_spec.instance_id, spec->ins_spec.major_version, spec->minor_version, code);
    } else if (ne_someip_find_status_pending == status) {
        printf("on_find_status: [0x%x:%d:%d:%d] ==> [PENDING], ret:[%d]\n", spec->ins_spec.service_id,
            spec->ins_spec.instance_id, spec->ins_spec.major_version, spec->minor_version, code);
    } else if (ne_someip_find_status_running == status) {
        printf("on_find_status: [0x%x:%d:%d:%d] ==> [RUNNING], ret:[%d]\n", spec->ins_spec.service_id,
            spec->ins_spec.instance_id, spec->ins_spec.major_version, spec->minor_version, code);
    }
}

void reg_find_service_handler(const ne_someip_required_service_instance_config_t* inst_config) {
    if (NULL == inst_config) {
        printf("inst_config null, error\n");
        return;
    }

    ne_someip_find_offer_service_spec_t spec1;
    spec1.ins_spec.service_id = inst_config->service_config->service_id;
    spec1.ins_spec.instance_id = inst_config->instance_id;
    spec1.ins_spec.major_version = inst_config->service_config->major_version;
    spec1.minor_version = inst_config->service_config->minor_version;
    ne_someip_client_context_reg_find_service_handler(g_client_context, &spec1, on_find_status, NULL);
}

void unreg_find_service_handler(const ne_someip_required_service_instance_config_t* inst_config) {
    if (NULL == inst_config) {
        printf("inst_config null, error\n");
        return;
    }

    ne_someip_find_offer_service_spec_t spec1;
    spec1.ins_spec.service_id = inst_config->service_config->service_id;
    spec1.ins_spec.instance_id = inst_config->instance_id;
    spec1.ins_spec.major_version = inst_config->service_config->major_version;
    spec1.minor_version = inst_config->service_config->minor_version;
    ne_someip_client_context_unreg_find_service_handler(g_client_context, &spec1, on_find_status);
}

void on_service_available(const ne_someip_find_offer_service_spec_t* spec, ne_someip_service_status_t status,
    void* user_data) {
    printf("on_service_available: [0x%x:%d:%d:%d] ===> [%s]\n", spec->ins_spec.service_id, spec->ins_spec.instance_id,
        spec->ins_spec.major_version, spec->minor_version,
        (ne_someip_service_status_available == status ? "available" : "unavailable"));
}

void reg_service_status_handler(const ne_someip_required_service_instance_config_t* inst_config) {
    if (NULL == inst_config) {
        printf("inst_config null, error\n");
        return;
    }

    ne_someip_find_offer_service_spec_t spec1;
    spec1.ins_spec.service_id = inst_config->service_config->service_id;
    spec1.ins_spec.instance_id = inst_config->instance_id;
    spec1.ins_spec.major_version = inst_config->service_config->major_version;
    spec1.minor_version = inst_config->service_config->minor_version;
    ne_someip_client_context_reg_service_status_handler(g_client_context, &spec1, on_service_available, NULL);
}

void unreg_service_status_handler(const ne_someip_required_service_instance_config_t* inst_config) {
    if (NULL == inst_config) {
        printf("inst_config null, error\n");
        return;
    }

    ne_someip_find_offer_service_spec_t spec1;
    spec1.ins_spec.service_id = inst_config->service_config->service_id;
    spec1.ins_spec.instance_id = inst_config->instance_id;
    spec1.ins_spec.major_version = inst_config->service_config->major_version;
    spec1.minor_version = inst_config->service_config->minor_version;
    ne_someip_client_context_unreg_service_status_handler(g_client_context, &spec1, on_service_available);
}

ne_someip_error_code_t start_find_service(const ne_someip_required_service_instance_config_t* inst_config) {
    if (NULL == inst_config) {
        printf("inst_config null, error\n");
        return ne_someip_error_code_failed;
    }

    ne_someip_find_offer_service_spec_t spec1;
    spec1.ins_spec.service_id = inst_config->service_config->service_id;
    spec1.ins_spec.instance_id = inst_config->instance_id;
    spec1.ins_spec.major_version = inst_config->service_config->major_version;
    spec1.minor_version = inst_config->service_config->minor_version;
    return ne_someip_client_context_start_find_service(g_client_context, &spec1, inst_config);
}

ne_someip_error_code_t stop_find_service(const ne_someip_required_service_instance_config_t* inst_config) {
    if (NULL == inst_config) {
        printf("inst_config null, error\n");
        return ne_someip_error_code_failed;
    }

    ne_someip_find_offer_service_spec_t spec1;
    spec1.ins_spec.service_id = inst_config->service_config->service_id;
    spec1.ins_spec.instance_id = inst_config->instance_id;
    spec1.ins_spec.major_version = inst_config->service_config->major_version;
    spec1.minor_version = inst_config->service_config->minor_version;
    return ne_someip_client_context_stop_find_service(g_client_context, &spec1);
}

void on_recv_response(ne_someip_required_service_instance_t* instance,
    ne_someip_header_t* header, ne_someip_payload_t* payload, void* user_data) {
    printf("on_recv_response: service_id:[%d], method_id:[%d], session_id:[%d], message_length:[%d]\n",
        header->service_id, header->method_id, header->session_id, header->message_length);

    uint32_t curCounter = 0;
    E2E_state_machine::E2ECheckStatus checkStatus = e2eCheck12289->check(header, payload, &curCounter, g_E2EState);
    printf("E2ECheck end: curCounter:[%d], g_send_counter_12289:[%d], checkStatus:[%d], E2EState:[%d], payload:%s\n",
        curCounter, g_send_counter_12289, checkStatus, g_E2EState,
        reinterpret_cast<char*>(payload->buffer_list[0]->data));
}

void on_send_req(ne_someip_required_service_instance_t* instance, void* seq_def,
    ne_someip_method_id_t method_id, ne_someip_error_code_t code, void* user_data) {
    // printf("on_send_req: method_id:[%d], ret:[0x%x]\n", method_id, code);
}

ne_someip_error_code_t create_and_send_req(const std::shared_ptr<NESomeIPE2EProtector>& e2eProtect) {
    ne_someip_header_t req_header;
    // ne_someip_method_id_t send_method_id = 12290;
    ne_someip_method_id_t send_method_id = 12289;
    ne_someip_error_code_t ret = ne_someip_client_context_create_req_header(g_req_instance, &req_header,
        send_method_id);
    if (ne_someip_error_code_ok != ret) {
        printf("client: ne_someip_client_context_create_req_header, ret:[0x%x]\n", ret);
        return ret;
    }

    ne_someip_payload_t* req_payload = ne_someip_payload_create();
    if (NULL == req_payload) {
        printf("ne_someip_payload_create return NULL.\n");
        return ne_someip_error_code_failed;
    }
    req_payload->num = 1;
    req_payload->buffer_list = reinterpret_cast<ne_someip_payload_slice_t**>(calloc(1,
        sizeof(*req_payload->buffer_list)));
    if (NULL == req_payload->buffer_list) {
        printf("calloc error.\n");
        return ne_someip_error_code_failed;
    }

    ne_someip_payload_slice_t* payload_slice = reinterpret_cast<ne_someip_payload_slice_t*>(calloc(1,
        sizeof(ne_someip_payload_slice_t)));
    if (NULL == payload_slice) {
        printf("calloc payload_slice error.\n");
        free(req_payload->buffer_list);
        return ne_someip_error_code_failed;
    }
    uint8_t tem_buffer[2] = "w";
    uint32_t payload_len = sizeof(tem_buffer);
    uint8_t* data = reinterpret_cast<uint8_t*>(calloc(1, payload_len));
    if (NULL == data) {
        printf("calloc data error.\n");
        free(data);
        return ne_someip_error_code_failed;
    }
    memcpy(data, tem_buffer, payload_len);
    payload_slice->free_pointer = data;
    payload_slice->data = data;
    payload_slice->length = payload_len;
    req_payload->buffer_list[0] = payload_slice;


    // e2e protect
    ret = e2eProtect->protect(&req_header, req_payload, &g_send_counter_12289);
    if (ne_someip_error_code_ok != ret) {
        printf("e2e protect error, ret:[0x%x]\n", ret);
        return ne_someip_error_code_failed;
    } else {
        printf("e2e protect success: counter:[%d], message_length:[%d]\n", g_send_counter_12289,
            req_header.message_length);
    }

    ret = ne_someip_client_context_send_request(g_client_context, g_req_instance, NULL, &req_header, req_payload);
    if (ne_someip_error_code_ok != ret) {
        printf("client: send method error, ret:[0x%x]\n", ret);
        return ne_someip_error_code_failed;
    }
    ne_someip_payload_unref(req_payload);

    return ne_someip_error_code_ok;
}

/* EOF */
