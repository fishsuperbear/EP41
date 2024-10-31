/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: custom_time_util.h
* @Date: 2023/02/14
* @Author: cheng
* @Desc: --
*/

#ifndef SERVICE_DATA_COLLECTION_COMMON_UTILS_CUSTOM_TIME_UTILS_H
#define SERVICE_DATA_COLLECTION_COMMON_UTILS_CUSTOM_TIME_UTILS_H

#include <time.h>
#include <string>

#include "advc-sdk/include/util/time_util.h"

namespace hozon {
namespace netaos {
namespace dc {

class CustomTimeUtil : public advc::Time {
   public:
    CustomTimeUtil(){};
    virtual time_t getLocalTime() override;
    virtual time_t getUnixTime() override;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif