/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_download.cpp
 * @Date: 2023/12/13
 * @Author: kun
 * @Desc: --
 */

#include "dc_download.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <memory>

#include "tsp_comm.h"
#include "json/json.h"
#include "utils/include/sign_utils.h"
#include "utils/include/path_utils.h"
#include "common/compress/include/dc_compress.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos::https;

Download::Download() {}

Download::~Download() {}

void Download::start() {
    DC_SERVER_LOG_DEBUG << "init tsp";
    TspComm::GetInstance().Init();
    tm = std::make_shared<TimerManager>();
    m_threadPoolFlex = std::make_shared<ThreadPoolFlex>(1);
    auto endTimer = tm->addTimer(TaskPriority::HIGH, 0, -1, 300000, [this] {
                            this->getCDNConfig();
                        });
    tm->start(*(m_threadPoolFlex.get()));
}

void Download::stop() {
    tm.reset();
    TspComm::GetInstance().Deinit();
}

void Download::getCDNConfig() {
    // getCarStrategies
    TspComm::HttpsParam httpsParam;
    httpsParam.method = 1;
    httpsParam.url = "http://vdc.autodrive.hozonauto.com:8081/api/strategyConfig/getCarStrategiesByVin";
    Json::Value getCarStrategiesJson;
    DC_SERVER_LOG_INFO << "begin to get remote config";
    getCarStrategiesJson["vin"] = TspComm::GetInstance().GetPkiVin();
    DC_SERVER_LOG_DEBUG << "vin: " << getCarStrategiesJson["vin"].asString();
    httpsParam.request_body = getCarStrategiesJson.toStyledString();
    httpsParam.headers["Content-Type"] = "application/json";
    std::future<TspComm::TspResponse> requsetResult = TspComm::GetInstance().RequestHttps(httpsParam);
    TspComm::TspResponse request = requsetResult.get();
    std::string triggerDynamicConfigStr = request.response;
    if (request.result_code != HttpsResult_Success) {
        DC_SERVER_LOG_ERROR << "https post trigger json failed, result code: " << request.result_code;
        return;
    }
    DC_SERVER_LOG_DEBUG << "triggerDynamicConfigStr: " << triggerDynamicConfigStr;
    Json::Reader reader;
	Json::Value rootValue;
    reader.parse(triggerDynamicConfigStr, rootValue, false);
    if (rootValue["code"].asInt64() != 200) {
        DC_SERVER_LOG_DEBUG << "trigger json get failed, code: " << rootValue["code"].asString();
        return;
    }
    if (m_triggerJsonMd5 != SignUtils::getMd5(triggerDynamicConfigStr)) {
        m_triggerJsonMd5 = SignUtils::getMd5(triggerDynamicConfigStr);
        // 更新trigger_limit.json
        DC_SERVER_LOG_DEBUG << "start write trigger json";
        SignUtils::WriteFileWithLock("/app/runtime_service/data_collection/conf/cdn_config.json", triggerDynamicConfigStr);
    } else {
        DC_SERVER_LOG_DEBUG << "trigger json not change";
    }
    std::vector<std::string> triggerList;
    if (rootValue["data"]["strategyData"]["strategies"].isArray()) {
        for (uint i = 0; i < rootValue["data"]["strategyData"]["strategies"].size(); i++) {
            if (rootValue["data"]["strategyData"]["strategies"][i]["cdnFlag"].asBool()) {
                DC_SERVER_LOG_DEBUG << "triggerId:" << rootValue["data"]["strategyData"]["strategies"][i]["triggerId"].asString();
                triggerList.push_back(rootValue["data"]["strategyData"]["strategies"][i]["triggerId"].asString());
            }
        }
    } else {
        DC_SERVER_LOG_DEBUG << "strategies is null";
        return;
    }
    if (triggerList.empty()) {
        DC_SERVER_LOG_DEBUG << "trigger list is empty";
    }
    httpsParam.url = "http://vdc.autodrive.hozonauto.com:8081/api/cdnConfig/getLatestConfigByTriggerId";
    Json::Value getLatestConfigByTriggerIdVec(Json::arrayValue);
    for (auto triggerId : triggerList) {
        getLatestConfigByTriggerIdVec.append(triggerId);
    }
    Json::Value getLatestConfigByTriggerIdJson;
    getLatestConfigByTriggerIdJson["triggerIdList"] = getLatestConfigByTriggerIdVec;
    httpsParam.request_body = getLatestConfigByTriggerIdJson.toStyledString();
    requsetResult = TspComm::GetInstance().RequestHttps(httpsParam);
    request = requsetResult.get();
    if (request.result_code != HttpsResult_Success) {
        DC_SERVER_LOG_ERROR << "https post cdn json failed, result code: " << request.result_code;
        return;
    }
    std::string cdnDynamicConfigStr = request.response;
    DC_SERVER_LOG_DEBUG << "cdnDynamicConfigStr: " << cdnDynamicConfigStr;
    reader.parse(cdnDynamicConfigStr, rootValue, false);
    if (rootValue["code"].asInt64() != 200) {
        DC_SERVER_LOG_DEBUG << "cdn json get failed, code: " << rootValue["code"].asString();
        return;
    }
    for (auto triggerId : triggerList) {
        for (auto ite = rootValue["data"][triggerId].begin(); ite != rootValue["data"][triggerId].end(); ite++) {
            TriggerStruct triggerStruct;
            triggerStruct.code = ite.key().asString();
            for (uint i = 0; i < rootValue["data"][triggerId][ite.key().asString()].size(); i++) {
                CdnStruct cdnStruct;
                cdnStruct.cdnPath = rootValue["data"][triggerId][ite.key().asString()][i]["cdnPath"].asString();
                cdnStruct.carPath = rootValue["data"][triggerId][ite.key().asString()][i]["carPath"].asString();
                cdnStruct.md5 = rootValue["data"][triggerId][ite.key().asString()][i]["md5"].asString();
                cdnStruct.code = rootValue["data"][triggerId][ite.key().asString()][i]["subBizCode"].asString();
                if (cdnStruct.cdnPath.find(".json") != std::string::npos) {
                    triggerStruct.versionCdnStruct = cdnStruct;
                } else if (cdnStruct.cdnPath.find(".om") != std::string::npos) {
                    triggerStruct.modelFileCdnStruct = cdnStruct;
                } else if (cdnStruct.cdnPath.find(".yaml") != std::string::npos) {
                    triggerStruct.modelCfgCdnStruct = cdnStruct;
                } else {
                    DC_SERVER_LOG_ERROR << "file type error: " << cdnStruct.cdnPath;
                }
            }
            downloadCdnFile(triggerId, triggerStruct);
        }
    }
}

void Download::downloadCdnFile(std::string triggerId, TriggerStruct triggerStruct) {
    for (uint i = 0; i < m_memoryTriggerStruct.size(); i++) {
        if (m_memoryTriggerStruct[i].code == triggerStruct.code) {
            if (m_memoryTriggerStruct[i].versionMD5 == triggerStruct.versionCdnStruct.md5) {
                DC_SERVER_LOG_DEBUG << "not need download cdn file: " << triggerStruct.versionCdnStruct.cdnPath;
            } else {
                downloadAndUpdate(m_memoryTriggerStruct[i], triggerId, triggerStruct);
            }
            return;
        }
    }
    MemoryTriggerStruct memoryTriggerStruct;
    downloadAndUpdate(memoryTriggerStruct, triggerId, triggerStruct);
    m_memoryTriggerStruct.push_back(memoryTriggerStruct);
}

void Download::downloadAndUpdate(MemoryTriggerStruct& memoryTriggerStruct, std::string triggerId, TriggerStruct triggerStruct) {
    std::string versionUrl = SignUtils::genSignUrl(triggerStruct.versionCdnStruct.cdnPath);
    DC_SERVER_LOG_DEBUG << "versionUrl: " << versionUrl;
    std::string versionFileName = triggerId + "-vehicle_data_collect_version.json";
    DC_SERVER_LOG_DEBUG << "versionFileName: " << versionFileName;
    std::string versionFolderPath;
    if (triggerStruct.versionCdnStruct.carPath.empty()) {
        versionFolderPath = "/opt/usr/col/perception";
    } else {
        versionFolderPath = triggerStruct.versionCdnStruct.carPath;
    }
    std::string versionFilePath = PathUtils::getFilePath(versionFolderPath, versionFileName);
    if (PathUtils::isFileExist(versionFilePath)) {
        std::string oldVersionData;
        SignUtils::ReadFileWithLock(versionFilePath, oldVersionData);
        if (SignUtils::getMd5(oldVersionData) == triggerStruct.versionCdnStruct.md5) {
            DC_SERVER_LOG_DEBUG << "not need download cdn file: " << triggerStruct.versionCdnStruct.cdnPath;
            return;
        }
    }
    TspComm::HttpsParam httpsParam;
    httpsParam.method = 0;
    httpsParam.url = versionUrl;
    std::future<TspComm::TspResponse> requsetResult = TspComm::GetInstance().RequestHttps(httpsParam);
    TspComm::TspResponse request = requsetResult.get();
    if (request.result_code != HttpsResult_Success) {
        DC_SERVER_LOG_ERROR << "https get cdn failed, result code: " << request.result_code;
        return;
    }
    std::string cdnFileData = request.response;
    DC_SERVER_LOG_DEBUG << "cdnFileData: " << cdnFileData;
    if (SignUtils::getMd5(cdnFileData) != triggerStruct.versionCdnStruct.md5) {
        DC_SERVER_LOG_ERROR << "file get fail, md5 error: " << triggerStruct.versionCdnStruct.cdnPath;
    }
    Json::Reader reader;
	Json::Value rootValue;
    reader.parse(cdnFileData, rootValue, false);
    std::string modelFileName = triggerId + "-" + rootValue["ModelFile"].asString();
    std::string ModelCfgName = triggerId + "-" + rootValue["ModelCfg"].asString();
    PathUtils::createFoldersIfNotExists(versionFolderPath);
    DC_SERVER_LOG_DEBUG  << "version file path: " << versionFilePath;
    SignUtils::WriteFileWithLock(versionFilePath, cdnFileData);
    if (memoryTriggerStruct.modelFileMD5 != triggerStruct.modelFileCdnStruct.md5) {
        httpsParam.url = SignUtils::genSignUrl(triggerStruct.modelFileCdnStruct.cdnPath);
        requsetResult = TspComm::GetInstance().RequestHttps(httpsParam);
        request = requsetResult.get();
        if (request.result_code != HttpsResult_Success) {
            DC_SERVER_LOG_ERROR << "https get cdn failed, result code: " << request.result_code;
            return;
        }
        cdnFileData = request.response;
        if (SignUtils::getMd5(cdnFileData) != triggerStruct.modelFileCdnStruct.md5) {
            DC_SERVER_LOG_ERROR << "file get fail, md5 error: " << triggerStruct.modelFileCdnStruct.cdnPath;
        }
        std::string modelFolderPath;
        if (triggerStruct.modelFileCdnStruct.carPath.empty()) {
            modelFolderPath = "/opt/usr/col/perception";
        } else {
            modelFolderPath = triggerStruct.modelFileCdnStruct.carPath;
        }
        std::string modelFilePath = PathUtils::getFilePath(modelFolderPath, modelFileName);
        PathUtils::createFoldersIfNotExists(modelFolderPath);
        DC_SERVER_LOG_DEBUG  << "model file path: " << modelFilePath;
        SignUtils::WriteFileWithLock(modelFilePath, cdnFileData);
        if (!memoryTriggerStruct.modelFilePath.empty() && memoryTriggerStruct.modelFilePath != modelFilePath) {
            PathUtils::removeFile(memoryTriggerStruct.modelFilePath);
        }
        memoryTriggerStruct.modelFileMD5 = triggerStruct.modelFileCdnStruct.md5;
        memoryTriggerStruct.modelFilePath = modelFilePath;
    } else {
        DC_SERVER_LOG_DEBUG << "not need download cdn file: " << triggerStruct.modelFileCdnStruct.cdnPath;
    }
    if (memoryTriggerStruct.modelCfgMD5 != triggerStruct.modelCfgCdnStruct.md5) {
        httpsParam.url = SignUtils::genSignUrl(triggerStruct.modelCfgCdnStruct.cdnPath);
        requsetResult = TspComm::GetInstance().RequestHttps(httpsParam);
        request = requsetResult.get();
        if (request.result_code != HttpsResult_Success) {
            DC_SERVER_LOG_ERROR << "https get cdn failed, result code: " << request.result_code;
            return;
        }
        cdnFileData = request.response;
        if (SignUtils::getMd5(cdnFileData) != triggerStruct.modelCfgCdnStruct.md5) {
            DC_SERVER_LOG_ERROR << "file get fail, md5 error: " << triggerStruct.modelCfgCdnStruct.cdnPath;
        }
        std::string modelCfgFolderPath;
        if (triggerStruct.modelCfgCdnStruct.carPath.empty()) {
            modelCfgFolderPath = "/opt/usr/col/perception";
        } else {
            modelCfgFolderPath = triggerStruct.modelCfgCdnStruct.carPath;
        }
        std::string modelCfgPath = PathUtils::getFilePath(modelCfgFolderPath, ModelCfgName);
        PathUtils::createFoldersIfNotExists(modelCfgFolderPath);
        DC_SERVER_LOG_DEBUG  << "model cfg file path: " << modelCfgPath;
        SignUtils::WriteFileWithLock(modelCfgPath, cdnFileData);
        if (!memoryTriggerStruct.modelCfgPath.empty() && memoryTriggerStruct.modelCfgPath != modelCfgPath) {
            PathUtils::removeFile(memoryTriggerStruct.modelCfgPath);
        }
        memoryTriggerStruct.modelCfgMD5 = triggerStruct.modelCfgCdnStruct.md5;
        memoryTriggerStruct.modelCfgPath = modelCfgPath;
    } else {
        DC_SERVER_LOG_DEBUG << "not need download cdn file: " << triggerStruct.modelCfgCdnStruct.cdnPath;
    }
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
