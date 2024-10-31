/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcu_log_collector.h
 * @Date: 2023/12/09
 * @Author: shenda
 * @Desc: --
 */

#pragma once

#include "collection/include/impl/log_collector.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos;

class MCULogCollector : public LogCollector {
   public:
    MCULogCollector();

    ~MCULogCollector() override;

    void active() override;

    bool getTaskResult(const std::string& type, struct DataTrans& dataStruct) override;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
