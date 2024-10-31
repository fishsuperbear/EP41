/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcap_h265_rw.cpp
 * @Date: 2023/10/16
 * @Author: kun
 * @Desc: --
 */

#include "service/data_collection/processor/include/impl/mcap_h265_rw.h"
#include "service/data_collection/processor/include/impl/desense_manager.h"
#include "utils/include/dc_logger.hpp"

namespace hozon {
namespace netaos {
namespace dc {

std::mutex McapH265RW::m_mtxForGetTaskName;
int McapH265RW::m_number = 0;

McapH265RW::McapH265RW() {
    m_taskStatus.store(TaskStatus::INITIAL, std::memory_order::memory_order_release);
    m_stopFlag = false;
}

McapH265RW::~McapH265RW() {}

void McapH265RW::onCondition(std::string type, char* data, Callback callback) {}

void McapH265RW::configure(std::string type, YAML::Node& node) {
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void McapH265RW::configure(std::string type, DataTrans& node) {
    m_inputH265FilePathVec.insert(m_inputH265FilePathVec.end(), node.pathsList[videoTopicMcapFiles].begin(), node.pathsList[videoTopicMcapFiles].end());
    m_node = node;
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void McapH265RW::active() {
    m_taskStatus.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    DC_SERVER_LOG_DEBUG << "mcap h265 rw: begin desense";

    std::lock_guard<std::mutex> lg(m_mtx);
    m_mtxForGetTaskName.lock();
    m_number++;
    std::string taskNumber = std::to_string(m_number);
    std::string taskName = m_node.memoryDatas[triggerIdData] + "_" + taskNumber;
    m_mtxForGetTaskName.unlock();
    DesenseManager::addDesenseTask(taskName, m_inputH265FilePathVec, 0);
    while (1) {
        if (DesenseManager::getDesenseTask(taskName, m_outputH265FilePathVec)) {
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    DC_SERVER_LOG_DEBUG << "mcap h265 rw: finish desense";
    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
}

void McapH265RW::deactive() {
    m_stopFlag = true;
}

TaskStatus McapH265RW::getStatus() {
    return m_taskStatus.load(std::memory_order::memory_order_acquire);
}

bool McapH265RW::getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) {
    if (m_taskStatus.load(std::memory_order::memory_order_acquire) != TaskStatus::FINISHED) {
        return false;
    } else {
        dataStruct.mergeDataStruct(m_node);
        dataStruct.memoryDatas[triggerIdData] = m_node.memoryDatas[triggerIdData];
        dataStruct.memoryDatas[triggerTime] = m_node.memoryDatas[triggerTime];
        dataStruct.pathsList[videoTopicMcapFiles].clear();
        dataStruct.pathsList[videoTopicMcapFiles].insert(m_outputH265FilePathVec.begin(), m_outputH265FilePathVec.end());
        return true;
    }
}

void McapH265RW::pause() {}

void McapH265RW::doWhenDestroy(const Callback& callback) {
    m_cb = callback;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
