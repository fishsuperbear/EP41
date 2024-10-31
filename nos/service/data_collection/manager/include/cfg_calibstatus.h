/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: cfg_calibstatus.h
 * @Date: 2023/12/26
 * @Author: liguoqiang
 * @Desc: --
 */

#pragma once
#ifndef DATA_COLLECTION_SERVICE_DATA_COLLECTION_MANAGER_INCLUDE_CFG_CALIBSTATUS_H__
#define DATA_COLLECTION_SERVICE_DATA_COLLECTION_MANAGER_INCLUDE_CFG_CALIBSTATUS_H__
#include <sys/stat.h>
#include <sys/types.h>

#include <bitset>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "config_param.h"
using namespace hozon::netaos::cfg;
namespace hozon {
namespace netaos {
namespace dc {
class CalibStatusReceive {
   public:
    CalibStatusReceive();
    ~CalibStatusReceive();
    void Init();
    void DeInit();
    struct calibStu {
        std::bitset<1> eol_bits;         // 产线标定
        std::bitset<3> static_as_bits;   // 静态售后标定
        std::bitset<3> dynamic_as_bits;  // 动态售后标定
        calibStu() {
            eol_bits.reset();
            static_as_bits.reset();
            dynamic_as_bits.reset();
        }
        void Clear() {
            eol_bits.reset();
            static_as_bits.reset();
            dynamic_as_bits.reset();
        }
    };

   private:
    void calibfunc(const std::string& clientname, const std::string& key, const uint8_t& value);
    std::string key_ = "system/calibrate_status";
    calibStu calibstu_;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // DATA_COLLECTION_SERVICE_DATA_COLLECTION_MANAGER_INCLUDE_CFG_CALIBSTATUS_H__
