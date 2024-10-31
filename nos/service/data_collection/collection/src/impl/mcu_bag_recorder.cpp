/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcu_log_collector.h
 * @Date: 2023/12/11
 * @Author: shenda
 * @Desc: --
 */

#include "collection/include/impl/mcu_bag_recorder.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos;

MCUBagRecorder::MCUBagRecorder() : BagRecorder() {
    DC_SERVER_LOG_INFO << "MCUBagRecorder: start";
}

MCUBagRecorder::~MCUBagRecorder() {
    DC_SERVER_LOG_INFO << "MCUBagRecorder: finish";
}

void MCUBagRecorder::active() {
    BagRecorder::active();
}

bool MCUBagRecorder::getTaskResult(const std::string& type, struct DataTrans& dataStruct) {
    auto this_ptr = this;
    std::string pointer_string((uint8_t *)&this_ptr, (uint8_t *)&this_ptr + sizeof(this_ptr));
    dataStruct.memoryDatas[MemoryDataType::MCUBagRecorderPointer] = pointer_string;
    DC_SERVER_LOG_INFO << "MCUBagRecorder: getTaskResult";
    dataStruct.dataType = DataTransType::memory;
    return true;
};

std::weak_ptr<bag::Recorder> MCUBagRecorder::getValidRecorder() {
    return std::weak_ptr<bag::Recorder>{rec_};
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
