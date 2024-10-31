/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: can parser abstract class
 */

#ifndef CAN_PARSER_H
#define CAN_PARSER_H

#include <vector>

#include <linux/can.h>
#include <linux/can/raw.h>

namespace hozon {
namespace netaos {
namespace canstack {

class CanParser {
   public:
    CanParser() = default;
    virtual ~CanParser() = default;

    virtual void Init() = 0;
    virtual void ParseCan(can_frame& receiveFrame) = 0;
    virtual void ParseCanfd(canfd_frame& receiveFrame) = 0;
    virtual void GetCanFilters(std::vector<can_filter>& filters){};
};

}  // namespace canstack   
}
}  // namespace hozon
#endif  // CAN_PARSER_H
