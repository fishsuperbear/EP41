/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcubag_record.h
 * @Date: 2023/11/22
 * @Author: shenda
 * @Desc: --
 */

#pragma once

#include <memory>
#include "collection/include/impl/bag_record.h"

namespace hozon {
namespace netaos {
namespace dc {

class MCUBagRecorder : public BagRecorder {
   public:
    MCUBagRecorder();
    ~MCUBagRecorder() override;
    void active() override;
    bool getTaskResult(const std::string& type, struct DataTrans& dataStruct) override;
    std::weak_ptr<bag::Recorder> getValidRecorder();

   private:
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
