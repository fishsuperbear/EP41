#include "someip_player_publisher.h"
#include <netinet/ip.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <thread>
#include "bag_data_type.h"
#include "data_tools_logger.hpp"
#include "ne_someip_daemon.h"

namespace hozon {
namespace netaos {
namespace bag {

// void run_daemon() {
//     ne_someip_daemon_t* daemon = ne_someip_daemon_init();
//     if (NULL == daemon) {
//         BAG_LOG_ERROR << "someip daemon init error";
//         exit(1);
//     }

//     while (true) {
//         sleep(3);
//     }
// }

uint16_t get_event_id(uint16_t method_id) {
    if (method_id >= 0x8000) {
        return method_id - 0x8000;
    }
    return method_id;
}

//  callback functions
void recv_subscribe_handler_callback(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* list, ne_someip_remote_client_info_t* client_info, void* user_data) {
    BAG_LOG_DEBUG << "++++++++++++++test_recv_subscribe_handler+++++++++++++++++";
    BAG_LOG_DEBUG << "recv subscribe, ipv4: " << client_info->ipv4 << " port: " << client_info->port;
}

void send_event_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id, ne_someip_error_code_t ret_code, void* user_data) {
    BAG_LOG_DEBUG << "++++++++++++++test_send_event_handler+++++++++++++++++";
    BAG_LOG_DEBUG << "send event: " << event_id << " ret code: " << ret_code;
}

void send_response_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id, ne_someip_error_code_t ret_code, void* user_data) {
    BAG_LOG_DEBUG << "++++++++++++++test_send_response_handler+++++++++++++++++";
}

void recv_request_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data) {
    BAG_LOG_DEBUG << "++++++++++++++test_recv_request_handler+++++++++++++++++";
}

void serivce_status_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status, ne_someip_error_code_t ret_code, void* user_data) {
    BAG_LOG_DEBUG << "++++++++++++++++++service status call back status [" << status << "]+++++++++++++++++++++";
}

SomeipPlayerPublisher::SomeipPlayerPublisher(const std::string& conf_path) : conf_path(conf_path) {
    // read someip config
    BAG_LOG_DEBUG << "load_config.";
    p_someip_config = load_config(conf_path);
    if (NULL == p_someip_config) {
        BAG_LOG_ERROR << "someip load config failed";
        return;
    }

    // create server context
    BAG_LOG_DEBUG << "server_create.";
    p_server_ctx = ne_someip_server_create(false, 0, 0);
    if (NULL == p_server_ctx) {
        BAG_LOG_ERROR << "someip create server failed";
        return;
    }

    // create instance
    // todo:
    BAG_LOG_DEBUG << "create provider instance.";
    for (size_t i = 0; i < p_someip_config->provided_instance_array.num; i++) {
        ne_someip_provided_instance_t* p_tmp_provider_ins = ne_someip_server_create_instance(p_server_ctx, &(p_someip_config->provided_instance_array.provided_instance_config[i]));
        if (NULL == p_tmp_provider_ins) {
            BAG_LOG_ERROR << "create provider instance falied";
            return;
        }
        provider_ins_list.push_back(p_tmp_provider_ins);
    }

    // p_provider_ins = ne_someip_server_create_instance(p_server_ctx, &(p_someip_config->provided_instance_array.provided_instance_config[0]));
    // if (NULL == p_provider_ins) {
    //     BAG_LOG_ERROR << "create provider instance falied";
    //     return;
    // }

    // ne_someip_server_reg_subscribe_handler(p_server_ctx, p_provider_ins, recv_subscribe_handler_callback, NULL);
    // ne_someip_server_reg_event_status_handler(p_server_ctx, p_provider_ins, send_event_handler_callback, NULL);
    std::vector<ne_someip_provided_instance_t*>::iterator it;
    it = provider_ins_list.begin();
    for (it = provider_ins_list.begin(); it != provider_ins_list.end(); it++) {
        // register hander
        BAG_LOG_DEBUG << "register hander.";
        ne_someip_server_reg_subscribe_handler(p_server_ctx, *it, recv_subscribe_handler_callback, NULL);
        ne_someip_server_reg_event_status_handler(p_server_ctx, *it, send_event_handler_callback, NULL);

        // offer
        BAG_LOG_DEBUG << "service offer.";
        ne_someip_server_offer(p_server_ctx, *it);
    }

    BAG_LOG_INFO << "someip player publisher create success!";
}

SomeipPlayerPublisher::~SomeipPlayerPublisher() {
    std::vector<ne_someip_provided_instance_t*>::iterator it;
    it = provider_ins_list.begin();
    for (it = provider_ins_list.begin(); it != provider_ins_list.end(); it++) {
        ne_someip_server_stop_offer(p_server_ctx, *it);

        ne_someip_server_unreg_subscribe_handler(p_server_ctx, *it);
        ne_someip_server_unreg_event_status_handler(p_server_ctx, *it);

        // destroy instance
        ne_someip_server_destroy_instance(p_server_ctx, *it);
    }

    ne_someip_server_unref(p_server_ctx);
    if (p_someip_config != nullptr) {
        ne_someip_config_release_someip_config(&p_someip_config);
    }
    // ne_someip_daemon_deinit();
    // p_provider_ins = NULL;
    p_server_ctx = NULL;
}

void SomeipPlayerPublisher::Publish(BagMessage* bagMessage) {
    // 解析要发送的数据
    std::string topic = bagMessage->topic;
    std::string type = bagMessage->type;
    unsigned char* p_message_data = bagMessage->data.m_payload->data;
    someip_message_t* p_someip_message = (someip_message_t*)p_message_data;
    someip_hdr_t someip_header = p_someip_message->someip_hdr;
    BAG_LOG_DEBUG << "message_id = " << someip_header.message_id.service_id << " + " << someip_header.message_id.method_id;
    BAG_LOG_DEBUG << "length = " << someip_header.length;
    BAG_LOG_DEBUG << "request_id = " << someip_header.request_id.client_id << " + " << someip_header.request_id.session_id;
    BAG_LOG_DEBUG << "protocol_version = " << someip_header.protocol_version;
    BAG_LOG_DEBUG << "interface_version = " << someip_header.interface_version;
    BAG_LOG_DEBUG << "msg_type = " << someip_header.msg_type;
    BAG_LOG_DEBUG << "return_code = " << someip_header.return_code;
    uint32_t someip_payload_length = p_someip_message->data_len;
    BAG_LOG_DEBUG << "someip_payload_length = " << someip_payload_length;
    unsigned char* p_payload = (unsigned char*)malloc(someip_payload_length);
    memcpy(p_payload, p_message_data + sizeof(someip_message_t), someip_payload_length);

    // 根据service_id获取p_provider_ins
    ne_someip_provided_instance_t* p_provider_ins = find_provider_ins(someip_header.message_id.service_id);
    if (NULL == p_provider_ins) {
        BAG_LOG_ERROR << "find provider instance fail! service id: " << someip_header.message_id.service_id;
        return;
    }

    // 构造ne_someip_payload_t
    ne_someip_payload_t* payload = ne_someip_payload_create();
    ne_someip_payload_slice_t* slice = (ne_someip_payload_slice_t*)malloc(sizeof(ne_someip_payload_slice_t));
    slice->length = someip_payload_length;
    slice->data = p_payload;
    slice->free_pointer = p_payload;
    payload->buffer_list = (ne_someip_payload_slice_t**)malloc(sizeof(ne_someip_payload_slice_t*));
    *(payload->buffer_list) = slice;
    payload->num = 1;
    ne_someip_header_t header;

    // 构造header
    ne_someip_server_create_notify_header(p_provider_ins, get_event_id(someip_header.message_id.method_id), &header);

    // 发送数据，引用计数-1
    auto status = ne_someip_server_send_event(p_server_ctx, p_provider_ins, nullptr, &header, payload);
    if (ne_someip_error_code_ok != status) {
        BAG_LOG_ERROR << "send event id " << get_event_id(someip_header.message_id.method_id) << " fail";
    }
    ne_someip_payload_unref(payload);
    BAG_LOG_DEBUG << "publish message: service id: " << someip_header.message_id.service_id << " method id: " << someip_header.message_id.method_id;
}

ne_someip_provided_instance_t* SomeipPlayerPublisher::find_provider_ins(uint16_t service_id) {
    for (size_t i = 0; i < p_someip_config->provided_instance_array.num; i++) {
        uint16_t tmp_service_id = p_someip_config->provided_instance_array.provided_instance_config[i].service_config->service_id;
        if (service_id == tmp_service_id) {
            BAG_LOG_DEBUG << "find find_provider_ins service id: " << service_id;
            return provider_ins_list[i];
        }
    }
    return NULL;
}

ne_someip_config_t* SomeipPlayerPublisher::load_config(const std::string someip_config_path) {
    std::ifstream ifs(someip_config_path, std::ios::binary);
    if (ifs.fail()) {
        BAG_LOG_ERROR << "Cannot open someip config file: " << someip_config_path;
        return nullptr;
    }

    ifs.seekg(0, std::ios::end);
    uint64_t length = ifs.tellg();
    ifs.seekg(0);
    const int buf_size = 102400;
    if (length >= buf_size) {
        BAG_LOG_ERROR << "someip config file too large: " << length;
        return nullptr;
    }
    char config_object[buf_size] = {0};
    ifs.read(config_object, buf_size);
    ifs.close();

    return ne_someip_config_parse_someip_config_by_content(config_object);
}

}  // namespace bag
}  // namespace netaos
}  // namespace hozon
