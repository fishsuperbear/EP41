/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: all_file_merge.cpp
 * @Date: 2023/11/23
 * @Author: kun
 * @Desc: --
 */

#include "processor/include/impl/all_file_merge.h"

#include <regex>
#include <cmath>

#include "middleware/tools/data_tools/mcap/include/merge.h"
#include "utils/include/path_utils.h"
#include "utils/include/trans_utils.h"
#include "openssl/sha.h"
#include "config_param.h"

using namespace hozon::netaos::mcap;
using namespace hozon::netaos::cfg;

namespace hozon {
namespace netaos {
namespace dc {

AllFileMerge::AllFileMerge() {
    m_taskStatus.store(TaskStatus::INITIAL, std::memory_order::memory_order_release);
    m_stopFlag = false;
}

AllFileMerge::~AllFileMerge() {}

void AllFileMerge::onCondition(std::string type, char* data, Callback callback) {}

void AllFileMerge::configure(std::string type, YAML::Node& node) {
    m_allFileMergeOption = node.as<AllFileMergeOption>();
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void AllFileMerge::configure(std::string type, DataTrans& node) {
    m_inputNode = node;
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void AllFileMerge::active() {
    m_taskStatus.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    DC_SERVER_LOG_DEBUG << "all file merge: begin";
    std::lock_guard<std::mutex> lg(m_mtx);
    if (m_allFileMergeOption.outputFolderPath.empty()) {
        DC_SERVER_LOG_WARN << "all file merge: output folder path is empty";
        m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
        return;
    }
    std::string outputFolderPath = PathUtils::getFilePath(m_allFileMergeOption.outputFolderPath, TransUtils::stringTransFileName("%Y%m%d-%H%M%S"));
    // 如果mcapFileVec中没有数据，则直接返回
    if (m_inputNode.pathsList[videoTopicMcapFiles].empty() && (m_inputNode.pathsList[commonTopicMcapFiles].empty())) {
        DC_SERVER_LOG_WARN << "all file merge: mcap file vector is empty";
        m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
        return;
    }
    int maxMcapFileCount = 0;
    if (m_inputNode.pathsList[videoTopicMcapFiles].size() > m_inputNode.pathsList[commonTopicMcapFiles].size()) {
        maxMcapFileCount = m_inputNode.pathsList[videoTopicMcapFiles].size();
    } else {
        maxMcapFileCount = m_inputNode.pathsList[commonTopicMcapFiles].size();
    }
    // 单个mcap数组范围最大为100，超过100部分忽略
    if (maxMcapFileCount > 100) {
        maxMcapFileCount = 100;
    }
    std::vector<std::string> videoTopicMcapFilesVec;
    for (const auto item : m_inputNode.pathsList[videoTopicMcapFiles]) {
        videoTopicMcapFilesVec.push_back(item);
    }
    std::vector<std::string> commonTopicMcapFilesVec;
    for (const auto item : m_inputNode.pathsList[commonTopicMcapFiles]) {
        commonTopicMcapFilesVec.push_back(item);
    }
    m_outputMcapFilePathVec.clear();
    int index = 1;
    std::string end;
    getUuid();
    for (int i = 0; i < maxMcapFileCount; i++) {
        if (m_stopFlag) {
            DC_SERVER_LOG_DEBUG << "all file merge: stop";
            m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
            return;
        }
        Merge merge;
        std::vector<std::string> mcapFilePathVec;
        std::string mergeFileName;
        if (i < commonTopicMcapFilesVec.size()) {
            mcapFilePathVec.push_back(commonTopicMcapFilesVec[i]);
            mergeFileName = PathUtils::getFileName(commonTopicMcapFilesVec[i]);
        }
        if (i < videoTopicMcapFilesVec.size()) {
            mcapFilePathVec.push_back(videoTopicMcapFilesVec[i]);
            if (mergeFileName.empty()) {
                mergeFileName = PathUtils::getFileName(videoTopicMcapFilesVec[i]);
            }
        }
        std::vector<std::string> attachmentStrNameVec;
        std::vector<std::string> attachmentStrVec;
        if (!m_inputNode.memoryDatas[hardwareVersionData].empty()) {
            attachmentStrNameVec.push_back("devm_version");
            attachmentStrVec.push_back(m_inputNode.memoryDatas[hardwareVersionData]);
        }
        if (!m_inputNode.memoryDatas[calibrationData].empty()) {
            attachmentStrNameVec.push_back("calibration.json");
            attachmentStrVec.push_back(m_inputNode.memoryDatas[calibrationData]);
        }
        if (!m_inputNode.memoryDatas[softwareVersionData].empty()) {
            attachmentStrNameVec.push_back("/app/version.json");
            attachmentStrVec.push_back(m_inputNode.memoryDatas[softwareVersionData]);
        }
        std::vector<std::string> attachmentFilePathVec;
        for (const auto item : m_inputNode.pathsList[calibrationFiles]) {
            attachmentFilePathVec.push_back(item);
        }
        for (const auto item : m_inputNode.pathsList[softwareVersionFiles]) {
            attachmentFilePathVec.push_back(item);
        }
        if (i == 0) {
            std::vector<std::string> firstAttachmentFilePathVec = attachmentFilePathVec;
            for (const auto item : m_inputNode.pathsList[faultManagerFiles]) {
                firstAttachmentFilePathVec.push_back(item);
            }
            merge.merge_mcap(mcapFilePathVec, attachmentStrNameVec, attachmentStrVec, firstAttachmentFilePathVec, PathUtils::getFilePath(outputFolderPath, mergeFileName));
        } else if (i == (maxMcapFileCount - 1)) {
            std::vector<std::string> lastAttachmentFilePathVec = attachmentFilePathVec;
            for (const auto item : m_inputNode.pathsList[hzLogFiles]) {
                lastAttachmentFilePathVec.push_back(item);
            }
            merge.merge_mcap(mcapFilePathVec, attachmentStrNameVec, attachmentStrVec, lastAttachmentFilePathVec, PathUtils::getFilePath(outputFolderPath, mergeFileName));
        } else {
            merge.merge_mcap(mcapFilePathVec, attachmentStrNameVec, attachmentStrVec, attachmentFilePathVec, PathUtils::getFilePath(outputFolderPath, mergeFileName));
        }
        std::set<std::string> mergeMcapFilePathVec = merge.get_output_file_path_vec();
        if (mergeMcapFilePathVec.empty()) {
            DC_SERVER_LOG_ERROR << "all file merge: mcap merge failed";
            for (std::string mcapFilePath : mcapFilePathVec) {
                DC_SERVER_LOG_ERROR << "all file merge: " << mcapFilePath;
            }
        } else {
            uint j = 0;
            for (std::string mergeMcapFilePath : mergeMcapFilePathVec) {
                j++;
                if ((j == mergeMcapFilePathVec.size()) && (i == (maxMcapFileCount - 1))) {
                    end = "end";
                }
                std::string outputMcapFilePath = changeFilePath(mergeMcapFilePath, index++, end);
                if (outputMcapFilePath.empty()) {
                    DC_SERVER_LOG_ERROR << "all file merge: mcap rename failed " << mergeMcapFilePath;
                } else {
                    m_outputMcapFilePathVec.push_back(outputMcapFilePath);
                }
            }
        }
    }
    changeTriggerTime();
    deleteOldMcapFile(videoTopicMcapFilesVec);
    deleteOldMcapFile(commonTopicMcapFilesVec);
    DC_SERVER_LOG_DEBUG << "all file merge: end";
    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
}

void AllFileMerge::deactive() {
    m_stopFlag = true;
}

TaskStatus AllFileMerge::getStatus() {
    return m_taskStatus.load(std::memory_order::memory_order_acquire);
}

bool AllFileMerge::getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) {
    if (m_taskStatus.load(std::memory_order::memory_order_acquire) != TaskStatus::FINISHED) {
        return false;
    } else {
        dataStruct.dataType = DataTransType::file;
        dataStruct.memoryDatas[triggerIdData] = m_inputNode.memoryDatas[triggerIdData];
        dataStruct.memoryDatas[triggerTime] = m_inputNode.memoryDatas[triggerTime];
        dataStruct.pathsList[commonTopicMcapFiles].clear();
        dataStruct.pathsList[commonTopicMcapFiles].insert(m_outputMcapFilePathVec.begin(), m_outputMcapFilePathVec.end());
        return true;
    }
}

void AllFileMerge::pause() {}

void AllFileMerge::doWhenDestroy(const Callback& callback) {
    m_cb = callback;
}

std::string AllFileMerge::changeFilePath(std::string oldFilePath, int index, std::string end) {
    std::string folderPath = PathUtils::getFolderName(oldFilePath);
    std::string timeStr;
    size_t pos = oldFilePath.find("EP41_ORIN_trigger_");
    if (pos != std::string::npos) {
        timeStr = oldFilePath.substr(pos + 18, 15);
    }
    std::string indexStr;
    if (index < 10) {
        indexStr = "0" + std::to_string(index);
    } else {
        indexStr = std::to_string(index);
    }

    std::string triggerId = m_inputNode.memoryDatas[triggerIdData];
    if (triggerId.empty()) {
        triggerId = "9999";
    }
    std::string newFilePath = m_uuid + "-EP41_ORIN_trigger-" + timeStr + "-" + triggerId + "-" + indexStr;
    if (end.empty()) {
        newFilePath = newFilePath + ".mcap";
    } else {
        newFilePath = newFilePath + "-" + end + ".mcap";
    }
    newFilePath = PathUtils::getFilePath(folderPath, newFilePath);
    if ((oldFilePath != newFilePath) && (PathUtils::isFileExist(oldFilePath)) && (!PathUtils::isFileExist(newFilePath))) {
        PathUtils::renameFile(oldFilePath, newFilePath);
        return newFilePath;
    } else {
        return "";
    }
}

void AllFileMerge::deleteOldMcapFile(std::vector<std::string> mcapFilePathVec) {
    for (std::string mcapFilePath : mcapFilePathVec) {
        if (PathUtils::isFileExist(mcapFilePath)) {
            PathUtils::removeFile(mcapFilePath);
        }
        std::string mcapFolderPath = PathUtils::getFolderName(mcapFilePath);
        std::smatch matchResult;
        std::tm tm{};
        std::string timeFormatOne = "\\d{6}-\\d{6}";
        std::regex regexPatternOne=std::regex(timeFormatOne);
        std::string timeFormatTwo = "\\d{4}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{6}";
        std::regex regexPatternTwo=std::regex(timeFormatTwo);
        if (std::regex_search(mcapFolderPath, matchResult, regexPatternOne)) {
            if (PathUtils::isDirExist(mcapFolderPath)) {
                PathUtils::removeFolder(mcapFolderPath);
            }
        } else if (std::regex_search(mcapFolderPath, matchResult, regexPatternTwo)) {
            if (PathUtils::isDirExist(mcapFolderPath)) {
                PathUtils::removeFolder(mcapFolderPath);
            }
        }
    }
}

void AllFileMerge::changeTriggerTime() {
    std::string triggerTime = m_inputNode.memoryDatas[MemoryDataType::triggerTime];
    int index = -1;
    int minDiff = 9999;
    for (uint i = 0; i < m_outputMcapFilePathVec.size(); i++) {
        size_t pos = m_outputMcapFilePathVec[i].find("EP41_ORIN_trigger-");
        std::string timeStr;
        if (pos != std::string::npos) {
            timeStr = m_outputMcapFilePathVec[i].substr(pos + 18, 15);
            if ((timeStr.size() == triggerTime.size()) && (timeStr.substr(0, 11) == triggerTime.substr(0, 11))) {
                int diff = abs(std::stoi(timeStr.substr(11, 4)) - std::stoi(triggerTime.substr(11, 4)));
                if (diff < minDiff) {
                    minDiff = diff;
                    index = i;
                }
            }
        }
    }
    if ((index != -1) && (minDiff != 0)) {
        std::string oldFilePath = m_outputMcapFilePathVec[index];
        size_t pos = oldFilePath.find("EP41_ORIN_trigger-");
        std::string filePathOne = oldFilePath.substr(0, pos + 18);
        std::string filePathTwo = triggerTime;
        std::string filePathThree = oldFilePath.substr(pos + 33, oldFilePath.length() - pos - 33);
        std::string newFilePath = filePathOne + filePathTwo + filePathThree;
        PathUtils::renameFile(oldFilePath, newFilePath);
        m_outputMcapFilePathVec[index] = newFilePath;
    }
}

void AllFileMerge::getUuid() {
    std::string vin;
    std::string timeStr = TransUtils::stringTransFileName("%Y%m%d-%H%M%S");
    auto cfgMgr = ConfigParam::Instance();
    cfgMgr->Init();
    auto res = cfgMgr->GetParam<std::string>("pki/vin", vin);
    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != res) {
        DC_LOG_ERROR << "all file merge: get config of vin error, res :" << res;
    }
    std::string data = vin + timeStr;
    SHA_CTX ctx;
    uint8_t sha[SHA_DIGEST_LENGTH] = {0};
    SHA1_Init(&ctx);
    SHA1(reinterpret_cast<const unsigned char*>(data.c_str()), data.size(), sha);
    std::stringstream ss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(sha[i]);
    }
    std::string sha_hex = ss.str();
    std::string uuid(sha_hex, 0, 16);
    m_uuid = uuid;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
