
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <json/json.h>
#include <vector>
#include <unordered_map>

#include "diag/common/include/data_def.h"

namespace hozon {
namespace netaos {
namespace diag {

void ParseJson(bool &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::string &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(int32_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(uint8_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(uint16_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(uint32_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(uint64_t &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(float &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(double &elem, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::vector<uint8_t> &vec, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::vector<uint16_t> &vec, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::vector<std::string> &vec, const Json::Value &currentNode, const std::string &key);

void ParseJson(DiagConfigInfo &configInfo, const Json::Value &currentNode);

void ParseJson(std::vector<DiagSubFuncDataInfo> &vec, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::vector<DiagRidSubFuncDataInfo> &vec, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint8_t, DiagSoftWareClusterDataInfo> &softWareClusterInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint8_t, DiagExternalServiceConfigDataInfo> &externalServiceConfigInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint8_t, DiagTransferConfigDataInfo> &transferConfigInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint8_t, DiagSessionDataInfo> &sessionInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint8_t, DiagSecurityLevelDataInfo> &securityLevelInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint8_t, DiagAccessPermissionDataInfo> &accessPermissionInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint8_t, DiagSidDataInfo> &sidInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint16_t, DiagDidDataInfo> &didInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint16_t, DiagRidDataInfo> &ridInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(std::unordered_map<uint32_t, DiagDtcDataInfo> &dtcInfoList, const Json::Value &currentNode, const std::string &key);

void ParseJson(DiagDemDataInfo& diagDemDataInfo, const Json::Value &currentNode, const std::string &key);

void UpdateJson(const bool &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const std::string &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const int32_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const uint32_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const uint8_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const uint16_t &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const float &elem, Json::Value &currentNode, const std::string &key);

void UpdateJson(const double &elem, Json::Value &currentNode, const std::string &key);

std::string ReadFile(const std::string &filePath);

ReadJsonFileReturnCode OpenJsonFile(const std::string &filePath, Json::Value &root);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, DiagConfigInfo &configInfo);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint8_t, DiagSoftWareClusterDataInfo> &softWareClusterInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint8_t, DiagExternalServiceConfigDataInfo> &externalServiceConfigInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint8_t, DiagTransferConfigDataInfo> &transferConfigInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint8_t, DiagSessionDataInfo> &sessionInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint8_t, DiagSecurityLevelDataInfo> &securityLevelInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint8_t, DiagAccessPermissionDataInfo> &accessPermissionInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint8_t, DiagSidDataInfo> &sidInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint16_t, DiagDidDataInfo> &didInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint16_t, DiagRidDataInfo> &ridInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint32_t, DiagDtcDataInfo> &dtcInfoList);

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, DiagDemDataInfo& diagDemDataInfo);

ReadJsonFileReturnCode WriteFile(const std::string &filePath, Json::Value &root);

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DATA_LOADER_H
