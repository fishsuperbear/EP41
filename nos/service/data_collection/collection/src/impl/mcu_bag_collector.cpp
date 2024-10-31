/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcu_log_collector.h
 * @Date: 2023/12/11
 * @Author: shenda
 * @Desc: --
 */

#include "collection/include/impl/mcu_bag_collector.h"
#include <chrono>
#include <thread>

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos;

MCUBagCollector::MCUBagCollector() : LogCollector() {
    DC_SERVER_LOG_DEBUG << "MCUBagCollector: start";
}

MCUBagCollector::~MCUBagCollector() {
    DC_SERVER_LOG_DEBUG << "MCUBagCollector: finish";
};

void MCUBagCollector::configure(std::string type, DataTrans& dataStruct) {
    if (dataStruct.memoryDatas.count(MemoryDataType::MCUBagRecorderPointer) > 0) {
        auto& pointer_string = dataStruct.memoryDatas[MemoryDataType::MCUBagRecorderPointer];
        auto& pMCUBagRecorder = *((MCUBagRecorder**)&pointer_string[0]);
        if (pMCUBagRecorder != nullptr) {
            m_pRecorder = pMCUBagRecorder->getValidRecorder();
        } else {
            DC_SERVER_LOG_ERROR << "MCUBagCollector: MCUBagRecorderPointer is null";
        }
    }
}

void MCUBagCollector::active() {
    DC_SERVER_LOG_INFO << "MCUBagCollector: active";
    try {
        if (auto pRecorder = m_pRecorder.lock(); pRecorder != nullptr) {
            pRecorder->SpliteBagNow();
            DC_SERVER_LOG_INFO << "MCUBagCollector: MCUBagRecorder split record done";
        } else {
            throw std::runtime_error("invalid MCUBagRecorder");
        }
    } catch (const std::exception& e) {
        DC_SERVER_LOG_ERROR << "MCUBagRecorder: split record error: " << e.what();
    }
    std::this_thread::sleep_for(std::chrono::seconds(2)); // wait for bag split completed
    LogCollector::active();
}

bool MCUBagCollector::getTaskResult(const std::string& type, struct DataTrans& dataStruct) {
    auto res = LogCollector::getTaskResult(type, dataStruct);
    for (const auto& elem : dataStruct.pathsList[hzLogFiles]) {
        DC_SERVER_LOG_DEBUG << "MCUBagCollector: getTaskResult " << elem;
    }
    return res;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
