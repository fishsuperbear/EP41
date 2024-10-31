/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: cfg_calibstatus.cpp
 * @Date: 2023/12/26
 * @Author: liguoqiang
 * @Desc: --
 */

#include "manager/include/cfg_calibstatus.h"
#include "pipeline/include/pipeline_manager.h"

namespace hozon {
namespace netaos {
namespace dc {


using namespace std;

CalibStatusReceive::CalibStatusReceive() { calibstu_.Clear(); }
CalibStatusReceive::~CalibStatusReceive() { calibstu_.Clear(); }

void CalibStatusReceive::calibfunc(const std::string& clientname, const std::string& key, const uint8_t& value) {
    std::cout << "calibfunc func receive the event that the value of param:test"
              << " is set to " << clientname << "  " << key << "  " << (int32_t)value << "  " << std::endl;
    // 0x01    EOL标定开始
    // 0x02    EOL标定结束
    // 0x03    行车camera售后静态标定开始
    // 0x04    行车camera售后静态标定结束
    // 0x05    行车camera售后动态标定开始
    // 0x06    行车camera售后动态标定结束
    // 0x07    泊车camera售后静态标定开始
    // 0x08    泊车camera售后静态标定结束
    // 0x09    泊车camera售后动态标定开始
    // 0x0a    泊车camera售后动态标定结束
    // 0x0b    Lidar售后静态标定开始
    // 0x0c    Lidar售后静态标定结束
    // 0x0d    Lidar售后动态标定开始
    // 0x0e    Lidar售后动态标定结束
    if (key_ == key) {
        switch (value) {
            case 0x01:
                calibstu_.eol_bits.reset(0);
                break;
            case 0x02:
                calibstu_.eol_bits.set(0);
                break;
            case 0x03:
                calibstu_.static_as_bits.reset(0);
                break;
            case 0x04:
                calibstu_.static_as_bits.set(0);
                break;
            case 0x05:
                calibstu_.dynamic_as_bits.reset(0);
                break;
            case 0x06:
                calibstu_.dynamic_as_bits.set(0);
                break;
            case 0x07:
                calibstu_.static_as_bits.reset(1);
                break;
            case 0x08:
                calibstu_.static_as_bits.set(1);
                break;
            case 0x09:
                calibstu_.dynamic_as_bits.reset(1);
                break;
            case 0x0a:
                calibstu_.dynamic_as_bits.set(1);
                break;
            case 0x0b:
                calibstu_.static_as_bits.reset(2);
                break;
            case 0x0c:
                calibstu_.static_as_bits.set(2);
                break;
            case 0x0d:
                calibstu_.dynamic_as_bits.reset(2);
                break;
            case 0x0e:
                calibstu_.dynamic_as_bits.set(2);
                break;
            default:
                std::cout << "invalid value " << value << std::endl;
                break;
        }
        //收到标定结束之后触发上传信号(不论是否收到开始信号)
        if (calibstu_.eol_bits.all()) {
            std::cout << "change mode EOL_CALIB " << std::endl;
            calibstu_.eol_bits.reset();
            PipelineManager::getInstance()->trigger("9001");
        }
        if (calibstu_.static_as_bits.all()) {
            std::cout << "change mode STATIC_AS_CALIB " << std::endl;
            calibstu_.static_as_bits.reset();
            PipelineManager::getInstance()->trigger("9002");
        }
        if (calibstu_.dynamic_as_bits.all()) {
            std::cout << "change mode DYNAMIC_AS_CALIB " << std::endl;
            calibstu_.dynamic_as_bits.reset();
            PipelineManager::getInstance()->trigger("9002");
        }
    }
}

void CalibStatusReceive::Init() {
    CfgResultCode res = ConfigParam::Instance()->Init(2000);
    std::cout << "Init  " << res << std::endl;
    if (res == CONFIG_OK) {
        ConfigParam::Instance()->MonitorParam<uint8_t>(key_, std::bind(&CalibStatusReceive::calibfunc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    }
}
void CalibStatusReceive::DeInit() {
    ConfigParam::Instance()->UnMonitorParam(key_);
    // CfgResultCode res = ConfigParam::Instance()->DeInit();
    // std::cout << "DeInit  " << res;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
