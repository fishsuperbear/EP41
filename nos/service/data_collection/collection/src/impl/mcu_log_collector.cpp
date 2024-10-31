/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcu_log_collector.h
 * @Date: 2023/12/11
 * @Author: shenda
 * @Desc: --
 */

#include "collection/include/impl/mcu_log_collector.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos;

MCULogCollector::MCULogCollector() : LogCollector() {
    DC_SERVER_LOG_DEBUG << "MCULogCollector: start";
}

MCULogCollector::~MCULogCollector() {
    DC_SERVER_LOG_DEBUG << "MCULogCollector: finish";
};

void MCULogCollector::active() {
    DC_SERVER_LOG_INFO << "MCULogCollector: active";
    LogCollector::active();
}

bool MCULogCollector::getTaskResult(const std::string& type, struct DataTrans& dataStruct) {
    auto res = LogCollector::getTaskResult(type, dataStruct);
    for (const auto& elem : dataStruct.pathsList[hzLogFiles]) {
        DC_SERVER_LOG_DEBUG << "MCULogCollector: getTaskResult " << elem;
    }
    return res;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
