/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: can_bag_collector.h
 * @Date: 2023/12/18
 * @Author: shenda
 * @Desc: --
 */

#pragma once

#include <memory>
#include "collection/include/impl/log_collector.h"

namespace hozon {
namespace netaos {
namespace dc {

class CANBagCollector : public LogCollector {
   public:
    CANBagCollector();

    ~CANBagCollector() override;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
