#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "json/json.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

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
            DG_ERROR << "DiagServer | Failed to open file: " << file;
            return "";
        }

        Json::CharReaderBuilder reader;
        Json::Value root;
        JSONCPP_STRING errs;
        bool res = Json::parseFromStream(reader, ifs, &root, &errs);
        if (!res || !errs.empty()) {
            DG_ERROR << "DiagServer | parseJson error, message: " << errs;
            return "";
        }
        ifs.close();

        Json::Value kv = root["kv_vec"];
        for (const auto& key_value : kv) {
            if (key == key_value["key"].asString()) {
                DG_INFO << "DiagServer | GetCfgValueFromFile read key: " << key_value["value"]["string"].asString();
                return key_value["value"]["string"].asString();
            }
        }

        return "";
    }
private:
    CfgValueInfo(){}
    ~CfgValueInfo(){}

};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon


