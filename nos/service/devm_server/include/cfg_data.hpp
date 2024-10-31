#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "json/json.h"
#include "devm_server_logger.h"


namespace hozon {
namespace netaos {
namespace devm_server {

class CfgValueInfo {
public:
    static CfgValueInfo* getInstance() {
        static CfgValueInfo instance;
        return &instance;
    }

    std::string GetCfgValueFromFile(std::string file, std::string key)
    {
        size_t pos = key.find('/');
        if (pos != std::string::npos) {
            key = key.substr(pos + 1);;
        }
        std::ifstream ifs(file);
        if (!ifs.is_open()) {
            DEVM_LOG_ERROR << "Failed to open file: " << file;
            return "";
        }

        Json::CharReaderBuilder reader;
        Json::Value root;
        JSONCPP_STRING errs;
        bool res = Json::parseFromStream(reader, ifs, &root, &errs);
        if (!res || !errs.empty()) {
            DEVM_LOG_ERROR << "parseJson error, message: " << errs;
            return "";
        }
        ifs.close();

        Json::Value kv = root["kv_vec"];
        for (const auto& key_value : kv) {
            if (key == key_value["key"].asString()) {
                DEVM_LOG_INFO << "GetCfgValueFromFile read key: " << key_value["value"]["string"].asString();
                return key_value["value"]["string"].asString();
            }
        }

        return "";
    }
private:
    CfgValueInfo(){}
    ~CfgValueInfo(){}

};

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon


