/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcu_bag_collector.h
 * @Date: 2023/11/21
 * @Author: shenda
 * @Desc: --
 */

#pragma once

#include <memory>
#include "collection/include/impl/log_collector.h"
#include "collection/include/impl/mcu_bag_recorder.h"

namespace hozon {
namespace netaos {
namespace dc {

class MCUBagCollector : public LogCollector {
   public:
    MCUBagCollector();

    ~MCUBagCollector() override;

    void configure(std::string type, DataTrans& node) override;

    void active() override;

    bool getTaskResult(const std::string& type, struct DataTrans& dataStruct) override;

   private:
    std::weak_ptr<hozon::netaos::bag::Recorder> m_pRecorder;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
