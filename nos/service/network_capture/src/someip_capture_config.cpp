#include <filesystem>

#include "network_capture/include/someip_capture_config.h"
#include "network_capture/include/someip_struct_define.h"
#include "json/json.h"


namespace hozon {
namespace netaos {
namespace network_capture {

std::map<std::string, std::string> eth_name_map = {{"NEP_SOC_1", "mgbe3_0.90"}, {"NEP_ADCS", "mgbe3_0"}};

std::unique_ptr<std::vector<std::unique_ptr<SomeipFilterInfo>>> SomeipFilterInfo::LoadConfig(std::string file_path) {

    std::map<std::string, SomeipFilterInfo> config_map;
    std::vector<std::string> jsonFiles;

    for (const auto& entry : std::filesystem::directory_iterator(file_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            jsonFiles.push_back(entry.path().string());
        }
    }

    // 打印所有.json文件的路径
    for (const auto& jsonFile : jsonFiles) {
        if (std::string::npos == jsonFile.find("_manifest")) continue;
        NETWORK_LOG_DEBUG << "config file path : " << jsonFile;

        if (0 == access(jsonFile.c_str(), F_OK)) {
            Json::Value root;
            Json::CharReaderBuilder readBuilder;
            std::ifstream ifs(jsonFile);
            std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
            JSONCPP_STRING errs;
            if (Json::parseFromStream(readBuilder, ifs, &root, &errs)) {
                for (const auto& filter : root["binding_list"]) {
                    std::string eth_name;
                    for (const auto& network_endpoint : filter["someip_services"]["network_endpoints"]) {
                        eth_name = eth_name_map[network_endpoint["network_id"].asString()];
                        config_map[eth_name].eth_name = eth_name;
                        config_map[eth_name].src_host = network_endpoint["ip_addr"].asString();
                    }
                    uint16_t port;
                    std::string instance;
                    Json::Value instances = filter["someip_services"]["required_instances"];
                    if(instances.empty())
                        instances = filter["someip_services"]["provided_instances"];
                    for (const auto& instance : instances) {
                        port = instance["udp_port"].asUInt();
                        if (port == 0) 
                            port = instance["tcp_port"].asUInt();
                        NETWORK_LOG_DEBUG << "port : " << port;
                        config_map[eth_name].ports.emplace_back(port);
                    }
                    instance = instances[0]["instance"].asString();
                    for (const auto& service : filter["someip_services"]["services"]) {
                        std::uint16_t service_id = service["service"].asUInt();
                        for (const auto& event : service["events"]) {
                            std::uint16_t method_id = event["event"].asUInt();
                            std::string topic_name = "/someip/" + service["short_name"].asString() + "/" + event["short_name"].asString() + "/" + instance;
                            config_map[eth_name].topic_map[MessageID(service_id, method_id) | 0x8000] = topic_name;
                        }
                        // for (const auto& event : service["events"]) {
                        //     std::uint16_t method_id = event["fields"].asUInt();
                        //     std::string topic_name = "/someip/" + service["short_name"].asString() + "/" + event["short_name"].asString() + "/" + instance;
                        //     config_map[eth_name].topic_map[MessageID(service_id, method_id)] = topic_name;
                        // }
                        // for (const auto& event : service["methods"]) {
                        //     std::uint16_t method_id = event["event"].asUInt();
                        //     std::string topic_name = "/someip/" + service["short_name"].asString() + "/" + event["short_name"].asString() + "/" + instance;
                        //     config_map[eth_name].topic_map[MessageID(service_id, method_id)] = topic_name;
                        // }
                    }
                }
            }
        }
    }


    // std::ifstream file(file_path);
    // Json::Value root;
    // file >> root;

    auto cfg_ptr = std::make_unique<std::vector<std::unique_ptr<SomeipFilterInfo>>>();
    NETWORK_LOG_DEBUG << "config_map.size() : " << config_map.size();
    for (const auto& config : config_map) {
        cfg_ptr->emplace_back(std::make_unique<SomeipFilterInfo>(config.second));
    }

    for (const auto& config : *cfg_ptr) {
        for (const auto& topic : config->topic_map) {
        NETWORK_LOG_DEBUG << "service_id : " << (topic.first >> 16) << " ,method_id : " << (topic.first & 0xFFFF) <<
                     " ,topic_name : " << topic.second;
        }
    }
    // for (const auto& filter : root["filter"]) {
    //     auto info = std::make_unique<SomeipFilterInfo>();
    //     info->eth_name = filter["eth_name"].asString();

    //     for (const auto& topic : filter["topic"]) {
    //         std::uint16_t service_id = topic["service_id"].asUInt();
    //         std::uint16_t method_id = topic["method_id"].asUInt();
    //         std::string topic_name = topic["topic_name"].asString();
    //         info->topic_map[MessageID(service_id, method_id)] = topic_name;
    //         std::cout << "service_id : " << topic["service_id"].asUInt() << " ,method_id : " << topic["method_id"].asUInt() <<
    //                      " ,topic_name : " << topic["topic_name"].asString() << std::endl;
    //     }

    //     for (const auto& port : filter["port"]) {
    //         info->ports.emplace_back(port.asUInt());
    //         // std::cout << "port : " << port.asUInt() << std::endl;
    //     }

    //     cfg_ptr->emplace_back(std::move(info));
    // }

    return cfg_ptr;
}
}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon