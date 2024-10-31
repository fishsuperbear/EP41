#pragma once
#include <vector>
#include "bag_message.hpp"
#include "impl/packet-someip.h"
#include "ne_someip_config_parse.h"
#include "ne_someip_server_context.h"

namespace hozon {
namespace netaos {
namespace bag {

class SomeipPlayerPublisher {
   public:
    SomeipPlayerPublisher(const std::string& conf_path);
    ~SomeipPlayerPublisher();
    // std::map<std::string, DataWriter*> _topicWriterMap;
    // bool prepareWriters(std::map<std::string, std::string> topicTypeMap);
    void Publish(BagMessage* bagMessage);

   private:
    // ne_someip_daemon_t* daemon = NULL;
    ne_someip_config_t* p_someip_config = NULL;
    ne_someip_server_context_t* p_server_ctx = NULL;
    // ne_someip_provided_instance_t* p_provider_ins = NULL;
    std::vector<ne_someip_provided_instance_t*> provider_ins_list;
    std::string conf_path = "./conf/someip_config.json";

    ne_someip_config_t* load_config(const std::string someip_config_path);
    ne_someip_provided_instance_t* find_provider_ins(uint16_t service_id);
    // void recv_subscribe_handler_callback(ne_someip_provided_instance_t* instance, const ne_someip_eventgroup_id_list_t* list, ne_someip_remote_client_info_t* client_info, void* user_data);
    // void send_event_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_event_id_t event_id, ne_someip_error_code_t ret_code, void* user_data);
    // void send_response_handler_callback(ne_someip_provided_instance_t* instance, void* seq_id, ne_someip_method_id_t method_id, ne_someip_error_code_t ret_code, void* user_data);
    // void recv_request_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_header_t* header, ne_someip_payload_t* payload, ne_someip_remote_client_info_t* client_info, void* user_data);
    // void serivce_status_handler_callback(ne_someip_provided_instance_t* instance, ne_someip_offer_status_t status, ne_someip_error_code_t ret_code, void* user_data);
};

}  // namespace bag
}  // namespace netaos
}  // namespace hozon