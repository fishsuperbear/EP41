/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcap_changer.cpp
 * @Date: 2023/09/25
 * @Author: kun
 * @Desc: --
 */
#include "service/data_collection/processor/include/impl/mcap_changer.h"
#include "utils/include/path_utils.h"
#include "utils/include/trans_utils.h"
#include "utils/include/dc_logger.hpp"

namespace hozon {
namespace netaos {
namespace dc {

const int MC_COMMON_TPC_INX = 0;
const int MC_VIDEO_TPC_INX = 1;

McapChanger::McapChanger() {
    m_taskStatus.store(TaskStatus::INITIAL, std::memory_order::memory_order_release);
    m_stopFlag = false;
}

McapChanger::~McapChanger() {}

void McapChanger::onCondition(std::string type, char* data, Callback callback) {}

void McapChanger::configure(std::string type, YAML::Node& node) {
    if (!node["method"] || !node["method"].IsScalar()) {
        DC_SERVER_LOG_ERROR << "mcap changer: mcap change method missed";
        m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
        return;
    }
    m_methodType = node["method"].as<std::string>();
    if (m_methodType == "filter") {
        m_filterOption = node.as<FilterOption>();
    } else if (m_methodType == "merge") {
        m_mergeOption = node.as<MergeOption>();
    } else if (m_methodType == "split") {
        m_splitOption = node.as<SplitOption>();
    } else {
        DC_SERVER_LOG_ERROR << "mcap changer: mcap change method error";
        m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
        return;
    }
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void McapChanger::configure(std::string type, DataTrans& node) {
    if (node.pathsList.empty()) {
        m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
    } else {
        m_inputMcapFilePathVec[MC_COMMON_TPC_INX] = node.pathsList[commonTopicMcapFiles];
        m_inputMcapFilePathVec[MC_VIDEO_TPC_INX] = node.pathsList[videoTopicMcapFiles];
        m_node = node;
        m_node.pathsList[commonTopicMcapFiles].clear();
        m_node.pathsList[videoTopicMcapFiles].clear();
        m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
    }
}

void McapChanger::active() {
    m_taskStatus.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    DC_SERVER_LOG_DEBUG << "mcap changer: begin";
    std::lock_guard<std::mutex> lg(m_mtx);
    if (m_inputMcapFilePathVec[MC_COMMON_TPC_INX].empty() && m_inputMcapFilePathVec[MC_VIDEO_TPC_INX].empty()) {
        DC_SERVER_LOG_WARN << "mcap changer: input mcap file is missed";
        m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
        return;
    }
    m_outputFilePathVec[MC_COMMON_TPC_INX].clear();
    m_outputFilePathVec[MC_VIDEO_TPC_INX].clear();
    std::string timestamp = TransUtils::stringTransFileName("%Y%m%d_%H%M%S");
    if (m_methodType == "filter") {
        DC_SERVER_LOG_DEBUG << "mcap changer: mcap filter begin";
        for (int i:{MC_COMMON_TPC_INX,MC_VIDEO_TPC_INX}) {
            std::string outputFolderPath = PathUtils::getFilePath(m_filterOption.outputPath, timestamp);
            for (std::string inputMcapFilePath : m_inputMcapFilePathVec[i]) {
                if (m_stopFlag) {
                    DC_SERVER_LOG_DEBUG << "mcap changer: stop";
                    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
                    return;
                }
                m_filter.filter_base_topic_list(inputMcapFilePath,
                                                m_filterOption.whiteTopicVec,
                                                m_filterOption.blackTopicVec,
                                                outputFolderPath);
                m_outputFilePathVec[i].insert(m_filter.get_output_file_path());
            }
        }
    } else if (m_methodType == "merge") {
        DC_SERVER_LOG_DEBUG << "mcap changer: mcap merge begin";
        for (int i:{MC_COMMON_TPC_INX,MC_VIDEO_TPC_INX}) {
            if (m_stopFlag) {
                DC_SERVER_LOG_DEBUG << "mcap changer: stop";
                m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
                return;
            }
            std::string outputFolderPath = PathUtils::getFilePath(m_mergeOption.outputPath, timestamp);
            std::vector<std::string> inputMcapFiles(m_inputMcapFilePathVec[i].begin(),m_inputMcapFilePathVec[i].end());
            m_merge.merge_mcap(inputMcapFiles,
                               m_mergeOption.attachmentFilePathVec,
                               PathUtils::getFilePath(outputFolderPath, "merge"));
            m_outputFilePathVec[i].insert(m_merge.get_output_file_path_vec().begin(), m_merge.get_output_file_path_vec().end());
        }
    } else if (m_methodType == "split") {
        DC_SERVER_LOG_DEBUG << "mcap changer: mcap split begin";
        for (int i:{MC_COMMON_TPC_INX,MC_VIDEO_TPC_INX}) {
            if (m_stopFlag) {
                DC_SERVER_LOG_DEBUG << "mcap changer: stop";
                m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
                return;
            }
            std::string outputFolderPath = PathUtils::getFilePath(m_splitOption.outputPath, timestamp);
            std::vector<std::string> inputMcapFiles(m_inputMcapFilePathVec[i].begin(),m_inputMcapFilePathVec[i].end());
            m_split.extract_attachments(inputMcapFiles,
                                        m_splitOption.attachmentFilePathVec,
                                        outputFolderPath);
            std::vector<std::string> splitResult = m_split.get_output_file_path_vec();
            m_outputFilePathVec[i].insert(splitResult.begin(), splitResult.end());
        }
    } else {
        DC_SERVER_LOG_ERROR << "mcap changer: mcap change method error";
        m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
        return;
    }
    DC_SERVER_LOG_DEBUG << "mcap changer: end";
    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
}

void McapChanger::deactive() {
    m_stopFlag = true;
}

TaskStatus McapChanger::getStatus() {
    return m_taskStatus.load(std::memory_order::memory_order_acquire);
}

bool McapChanger::getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) {
    if (m_taskStatus.load(std::memory_order::memory_order_acquire) != TaskStatus::FINISHED) {
        return false;
    } else {
        dataStruct.dataType = DataTransType::file;
        dataStruct.memoryDatas[triggerIdData] = m_node.memoryDatas[triggerIdData];
        dataStruct.memoryDatas[triggerTime] = m_node.memoryDatas[triggerTime];
        dataStruct.pathsList[commonTopicMcapFiles].insert(m_outputFilePathVec[MC_COMMON_TPC_INX].begin(), m_outputFilePathVec[MC_COMMON_TPC_INX].end());
        dataStruct.pathsList[videoTopicMcapFiles].insert(m_outputFilePathVec[MC_VIDEO_TPC_INX].begin(), m_outputFilePathVec[MC_VIDEO_TPC_INX].end());
        dataStruct.mergeDataStruct(m_node);
        return true;
    }
}

void McapChanger::pause() {

}

void McapChanger::doWhenDestroy(const Callback& callback) {
    m_cb = callback;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
