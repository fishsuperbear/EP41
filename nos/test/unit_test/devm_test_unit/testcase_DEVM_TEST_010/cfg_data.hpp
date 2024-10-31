#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "json/json.h"


namespace hozon {
namespace netaos {
namespace unit_test {

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
            std::cout << "Failed to open file: " << file << std::endl;
            return "";
        }

        Json::CharReaderBuilder reader;
        Json::Value root;
        JSONCPP_STRING errs;
        bool res = Json::parseFromStream(reader, ifs, &root, &errs);
        if (!res || !errs.empty()) {
            std::cout << "parseJson error, message: " << errs << std::endl;
            return "";
        }
        ifs.close();

        Json::Value kv = root["kv_vec"];
        for (const auto& key_value : kv) {
            if (key == key_value["key"].asString()) {
                std::cout << "GetCfgValueFromFile read key: " << key_value["value"]["string"].asString() << std::endl;
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


