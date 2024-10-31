/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: pipeline_manager.cpp
 * @Date: 2023/08/14
 * @Author: cheng
 * @Desc: --
 */

#include "pipeline/include/pipeline_manager.h"
#include "basic/retcode_define.h"
#include "processor/include/impl/compressor.h"
#include "utils/include/path_utils.h"
#include "utils/include/trans_utils.h"
#include "json/json.h"

namespace hozon {
namespace netaos {
namespace dc {
void PipelineManager::trigger(const std::shared_ptr<triggerInfo> &req, std::shared_ptr<triggerResult> &resp) {
    DC_OPER_LOG_INFO << "receive trigger,type:" << req->type() << ",clientName:" << req->clientName() << ",value:"
                       << req->value()<< ",priority:" << req->priority() << ",msg:" << req->msg();
    PipeLineTask plt;
    bool result;
    DataTrans dt;
    if (req->type()=="triggerdesc") {
        auto spliterIndex = req->value().find('|');
        auto triggerId = req->value().substr(0, spliterIndex);
        auto time =  req->value().substr(spliterIndex+1);
        dt.memoryDatas[triggerTime]=time;
        result = cfgm_->getTriggerPipeTask(req->value(), plt);
    } else {
        result = cfgm_->getTriggerPipeTask(req->value(), plt);
    }
    if (!result) {
        resp->msg("load trigger config failed for "+req->value());
        resp->retCode(RetCode::LOAD_TRIGGER_CONFIG_FAILED);
        return;
    }
    DC_SERVER_LOG_INFO << "task " << plt.taskName << " with priority " << plt.priority << " triggered";
    resp->retCode(RetCode::SUCCESS);
    resp->msg("OK");
    dt.memoryDatas[triggerIdData]=req->value();
    dt.memoryDatas[priorityData]=plt.priority;
    setPipelineTaskInput(plt.taskName, dt);
    addTask(plt);
}

void PipelineManager::triggerUpload(const std::shared_ptr<triggerUploadInfo> &req, std::shared_ptr<triggerResult> &resp) {
    DC_OPER_LOG_INFO << "receive trigger,type:" << req->type() << ",clientName:" << req->clientName();
    for (std::string path : req->pathList()) {
        DC_SERVER_LOG_INFO << ",filPath:" << path;
    }
    DC_OPER_LOG_INFO << ",fileType:" << req->fileType() << ",name:" << req->fileName() << ",cacheFileNum:" << req->cacheFileNum();
    PipeLineTask plt;
    std::lock_guard<std::mutex> lg(uploadMtx_);
    auto result = cfgm_->getPipeTaskFromName("uploadTrigger", plt);
    if (!result) {
        resp->msg("load trigger config failed for "+req->fileType());
        resp->retCode(RetCode::LOAD_TRIGGER_CONFIG_FAILED);
        return;
    }
     // 保存cache number
    std::string jsonFolderPath = "/opt/usr/col/json/" + req->clientName() + "/" + req->fileType();
    PathUtils::createFoldersIfNotExists(jsonFolderPath);
    std::string jsonFilePath = jsonFolderPath + "/config.json";
    if (PathUtils::isFileExist(jsonFilePath)) {
        Json::Value root;
        Json::CharReaderBuilder reader;
        JSONCPP_STRING errs;
        std::ifstream ifs(jsonFilePath);
        if (!ifs.is_open()) {
            DC_SERVER_LOG_ERROR << "Failed to open file: " << jsonFilePath;
            return;
        }
        bool res = Json::parseFromStream(reader, ifs, &root, &errs);
        if (!res || !errs.empty()) {
            DC_SERVER_LOG_ERROR << "parseJson error! ";
            return;
        }
        ifs.close();
        if (req->cacheFileNum() < root["maxFiles"].asInt()) {
            writeMaxFilesToJson(jsonFilePath, req->cacheFileNum());
        }
    } else {
        writeMaxFilesToJson(jsonFilePath, req->cacheFileNum());
    }

    plt.taskName = req->fileType()+"_"+TimeUtils::timestamp2ReadableStr(TimeUtils::getDataTimestamp());
    for (auto &taskInfo : plt.pipeLine) {
        if (taskInfo.type == PipeTaskType::destination) {
            std::string taskName = getManager(taskInfo.type)->getTaskName(taskInfo.type);
            taskInfo.taskName = taskName;
            auto taskInstance = getManager(taskInfo.type)->tryCreateTask(taskInfo.policy,taskName);
            YAML::Node fileTypeNode = YAML::Load("uploadType: " + req->fileType());
            taskInstance->configure("uploadType", fileTypeNode);
        }
    }

    std::vector<int> waitItems = {-1};
    std::string fakerPreTaskName=plt.taskName+"_pre";
    DataTrans dts;
    dts.dataType = DataTransType::fileAndFolder;
    dts.pathsList[uploadReqOnlyFiles].insert(req->pathList().begin(),req->pathList().end());
    dts.memoryDatas[uploadFileNameDefine] = TransUtils::stringTransFileName(req->fileName());
    dts.memoryDatas[uploadFileDeleteFlag] = std::to_string(req->deleteAfterCompress());
    resp->retCode(RetCode::SUCCESS);
    resp->msg("OK");
    setPipelineTaskInput(plt.taskName,dts);
    addTask(plt);
    DC_SERVER_LOG_INFO << "triggerUpload end";
}

void PipelineManager::writeMaxFilesToJson(std::string jsonFilePath, int maxFiles) {
    Json::Value root;
    Json::StyledStreamWriter writer;
    std::ofstream ofs(jsonFilePath);
    root["maxFiles"] = maxFiles;
    writer.write(ofs, root);
    ofs.close();
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
