/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: load data
 */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <unordered_map>
#include "json/json.h"
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {

void ParseJson(bool &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::string &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(int32_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(uint32_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(uint8_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(uint16_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(float &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(double &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint32_t, FaultLockInfo> &faultLockInfoList, const Json::Value &currentNode, const std::string &key);

ReadJsonFileReturnCode OpenJsonFile(const std::string &filePath, Json::Value &root);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint32_t, FaultLockInfo> &faultLockInfoList);

void UpdateJson(const bool &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const std::string &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const int32_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const uint32_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const uint8_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const uint16_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const float &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const double &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const std::vector<FaultLockInfo> &faultLockInfos, Json::Value &currentNode, const std::string &key);

ReadJsonFileReturnCode WriteJsonFile(const std::string &filePath, const std::vector<FaultLockInfo> &faultLockInfos);

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // DATA_LOADER_H
