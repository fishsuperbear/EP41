/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: can_log_collector.h
 * @Date: 2023/12/18
 * @Author: shenda
 * @Desc: --
 */

#include "collection/include/impl/can_bag_collector.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos;

CANBagCollector::CANBagCollector() : LogCollector() {
    DC_SERVER_LOG_DEBUG << "CANBagCollector: start";
}

CANBagCollector::~CANBagCollector() {
    DC_SERVER_LOG_DEBUG << "CANBagCollector: finish";
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
