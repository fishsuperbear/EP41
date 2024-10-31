/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: load data to fm
*/

#include <fstream>
#include <unistd.h>
#include <sys/param.h>
#include <iostream>

#include "phm_server/include/common/data_loader.h"

namespace hozon {
namespace netaos {
namespace phm_server {

void ParseJson(bool &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isBool()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [bool] type.");
        return;
    }
    elem = jsonValue.asBool();
}

void ParseJson(std::string &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isString()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [string] type.");
        return;
    }
    elem = static_cast<std::string>(jsonValue.asString());
}

void ParseJson(int32_t &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [int32] type.");
        return;
    }
    elem = static_cast<int32_t>(jsonValue.asInt());
}

void ParseJson(uint32_t &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isUInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [uint32] type.");
        return;
    }
    elem = static_cast<uint32_t>(jsonValue.asUInt());
}

void ParseJson(uint8_t &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isUInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [uint8] type.");
        return;
    }
    elem = static_cast<uint8_t>(jsonValue.asUInt());
}

void ParseJson(uint16_t &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isUInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [uint16] type.");
        return;
    }
    elem = static_cast<uint16_t>(jsonValue.asUInt());
}

void ParseJson(float &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isDouble()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [float] type.");
        return;
    }
    elem = static_cast<float>(jsonValue.asDouble());
}

void ParseJson(double &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }
    if (!jsonValue.isDouble()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [double] type.");
        return;
    }
    elem = jsonValue.asDouble();
}

void ParseJson(std::unordered_map<uint32_t, FaultLockInfo> &faultLockInfoList, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    FaultLockInfo faultLockInfo;
    faultLockInfo.isHandled = 0;
    for (Json::ArrayIndex i = 0; i < jsonValue.size(); ++i) {
        ParseJson(faultLockInfo.faultId, jsonValue[i], "FaultId");
        ParseJson(faultLockInfo.faultObj, jsonValue[i], "FaultObj");
        ParseJson(faultLockInfo.lockCount, jsonValue[i], "LockCount");
        ParseJson(faultLockInfo.recoverCount, jsonValue[i], "RecoverCount");
        ParseJson(faultLockInfo.faultToHMIData, jsonValue[i], "FaultToHMIData");
        ParseJson(faultLockInfo.lockFaultToHMIData, jsonValue[i], "LockFaultToHMIData");
        ParseJson(faultLockInfo.isBlockedFault, jsonValue[i], "IsBlockedFault");
        ParseJson(faultLockInfo.faultCount, jsonValue[i], "FaultCount");
        ParseJson(faultLockInfo.lockedNumber, jsonValue[i], "LockedNumber");
        ParseJson(faultLockInfo.faultRecoverCount, jsonValue[i], "FaultRecoverCount");
        ParseJson(faultLockInfo.isNeedToRecover, jsonValue[i], "IsNeedToRecover");

        if (faultLockInfo.lockedNumber) {
            faultLockInfo.faultRecoverCount++;
        }

        if (faultLockInfo.faultRecoverCount > faultLockInfo.recoverCount) {
            faultLockInfo.isNeedToRecover = 1;
        }

        uint32_t faultKey = faultLockInfo.faultId * 100 + faultLockInfo.faultObj;
        faultLockInfoList.insert(std::make_pair(faultKey, faultLockInfo));
    }
}

std::string ReadFile(const std::string &filePath)
{
    char absolutePath[PATH_MAX] {};
    char *ret = realpath(filePath.data(), absolutePath);
    if (ret == nullptr) {
        return "";
    }
    std::ifstream in(absolutePath);
    if (!in.good()) {
        return "";
    }
    std::string contents { std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
    return contents;
}

ReadJsonFileReturnCode OpenJsonFile(const std::string &filePath, Json::Value &root)
{
    std::string fileContent = ReadFile(filePath);
    if (fileContent.empty()) {
        // PHM_LOG_ERROR << "Failed to read the json file.";
        return ERROR;
    }
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    const int rawJsonLength = static_cast<int>(fileContent.length());
    std::string err;
    if (!reader->parse(fileContent.c_str(), fileContent.c_str() + rawJsonLength, &root, &err)) {
        // PHM_LOG_ERROR << ("Failed to parse the json file."
        //     "\nThe error message for parsing the json file is: \n" +
        //     err);
        return ERROR;
    }

    return OK;
}

ReadJsonFileReturnCode ReadJsonFile(const std::string &filePath, std::unordered_map<uint32_t, FaultLockInfo> &faultLockInfoList)
{
    Json::Value root;
    auto ret = OpenJsonFile(filePath, root);
    if (ret != OK) {
        return ret;
    }

    ParseJson(faultLockInfoList, root, "FaultLockList");
    return OK;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// update and write
void UpdateJson(const bool &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isBool()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [bool] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const std::string &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isString()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [string] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const int32_t &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [int32] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const uint32_t &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isUInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [uint32] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const uint8_t &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isUInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [uint8] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const uint16_t &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isUInt()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [uint16] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const float &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isDouble()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [float] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const double &elem, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isDouble()) {
        std::cout << ("Failed to convert the value of the " + key + " key to the [double] type.");
        return;
    }

    currentNode[key] = elem;
}

void UpdateJson(const std::vector<FaultLockInfo> &faultLockInfos, Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    uint32_t size = jsonValue.size();
    // std::cout << "FaultLock::RecordTask UpdateJson size: "  << size << " faultLockInfos.size:" << faultLockInfos.size() << std::endl;
    if(size != faultLockInfos.size()) {
        std::cout << "FaultLock::RecordTask UpdateJson data error."  << std::endl;
        return;
    }

    for (Json::ArrayIndex i = 0; i < size; ++i) {
        UpdateJson(faultLockInfos[i].faultId,currentNode[key][i], "FaultId");
        UpdateJson(faultLockInfos[i].faultObj, currentNode[key][i], "FaultObj");
        UpdateJson(faultLockInfos[i].lockCount, currentNode[key][i], "LockCount");
        UpdateJson(faultLockInfos[i].recoverCount, currentNode[key][i], "RecoverCount");
        UpdateJson(faultLockInfos[i].faultToHMIData, currentNode[key][i], "FaultToHMIData");
        UpdateJson(faultLockInfos[i].lockFaultToHMIData, currentNode[key][i], "LockFaultToHMIData");
        UpdateJson(faultLockInfos[i].isBlockedFault, currentNode[key][i], "IsBlockedFault");
        UpdateJson(faultLockInfos[i].faultCount, currentNode[key][i], "FaultCount");
        UpdateJson(faultLockInfos[i].isHandled, currentNode[key][i], "IsHandled");
        UpdateJson(faultLockInfos[i].lockedNumber, currentNode[key][i], "LockedNumber");
        UpdateJson(faultLockInfos[i].faultRecoverCount, currentNode[key][i], "FaultRecoverCount");
        UpdateJson(faultLockInfos[i].isNeedToRecover, currentNode[key][i], "IsNeedToRecover");
    }
}

ReadJsonFileReturnCode UpdateJsonFile(const std::string &filePath, Json::Value &root)
{
    std::ofstream out(filePath, std::ios::in | std::ios::out);
    if(!out.good())
    {
        return ERROR;
    }

    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &out);
    return OK;
}

ReadJsonFileReturnCode WriteJsonFile(const std::string &filePath, const std::vector<FaultLockInfo> &faultLockInfos)
{
    Json::Value root;
    auto ret = OpenJsonFile(filePath, root);
    if (ret != OK) {
        return ret;
    }

    Json::Value currentNode;
    UpdateJson(faultLockInfos, root, "FaultLockList");

    ret = UpdateJsonFile(filePath, root);
    if (ret != OK) {
        return ret;
    }

    return OK;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
