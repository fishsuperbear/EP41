/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault event inhibit
*/
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"
#include "phm_server/include/fault_manager/manager/phm_config_monitor.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/time_manager.h"
#include "config_param.h"
#include <cstdint>
#include <iostream>

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace hozon::netaos::cfg;
enum PHM_INHIBIT_TYPE
{
    PHM_INHIBIT_TYPE_NONE = 0,
    PHM_INHIBIT_TYPE_OTA,
    PHM_INHIBIT_TYPE_CALIBRATION,
    PHM_INHIBIT_TYPE_PARKING,
    PHM_INHIBIT_TYPE_85,
    PHM_INHIBIT_TYPE_POWERMODE_OFF,
    PHM_INHIBIT_TYPE_RUNNING_MODE
};


PhmCfgMonitor* PhmCfgMonitor::instance_ = nullptr;
std::mutex PhmCfgMonitor::mtx_;
const int32_t INHIBIT_TIME = 2500;
static uint8_t m_lastMode = 0;
static bool bRecoverFault = false; // recover parking's XXX faults
int timerFd_ = -1;
std::shared_ptr<TimerManager> time_mgr_ = nullptr;
std::thread monitor_thread_;

PhmCfgMonitor*
PhmCfgMonitor::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new PhmCfgMonitor();
        }
    }

    return instance_;
}

PhmCfgMonitor::PhmCfgMonitor()
{
    PHMS_INFO << "PhmCfgMonitor::PhmCfgMonitor";
    time_mgr_.reset(new TimerManager());
}

PhmCfgMonitor::~PhmCfgMonitor()
{
    PHMS_INFO << "PhmCfgMonitor::~PhmCfgMonitor";
}

void
PhmCfgMonitor::Init()
{
    PHMS_INFO << "PhmCfgMonitor::Init";
    if (time_mgr_ != nullptr) {
        time_mgr_->Init();
    }

    auto f = [&](){
        hozon::netaos::cfg::CfgResultCode res = ConfigParam::Instance()->Init(3000);
        if (CONFIG_OK != res) {
            PHMS_INFO << "configparam init error:";
        }

        res = CONFIG_TIME_OUT;
        static uint32_t count = 0;
        while (CONFIG_OK != res && count++ < 25) {
            PHMS_DEBUG << "PhmCfgMonitor::Init config param 'power_mode' wating......";
            res = ConfigParam::Instance()->MonitorParam<uint8_t>("system/power_mode", std::bind(&PhmCfgMonitor::powerModeCb,
                this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (count >= 25) {
            PHMS_WARN << "PhmCfgMonitor::Init config param 'power_mode' failed";
        }
        else {
            PHMS_INFO << "PhmCfgMonitor::Init config param 'power_mode' success";
        }

        res = CONFIG_TIME_OUT;
        count = 0;
        while (CONFIG_OK != res && count++ < 25) {
            PHMS_DEBUG << "PhmCfgMonitor::Init config param 'calibrate_status' wating......";
            res = ConfigParam::Instance()->MonitorParam<uint8_t>("system/calibrate_status", std::bind(&PhmCfgMonitor::calibrateStatusCb,
                this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (count >= 25) {
            PHMS_WARN << "PhmCfgMonitor::Init config param 'calibrate_status' failed";
        }
        else {
            PHMS_INFO << "PhmCfgMonitor::Init config param 'calibrate_status' success";
        }

        res = CONFIG_TIME_OUT;
        count = 0;
        while (CONFIG_OK != res && count++ < 25) {
            PHMS_DEBUG << "PhmCfgMonitor::Init config param '85_status' wating......";
            res = ConfigParam::Instance()->MonitorParam<uint8_t>("system/85_status", std::bind(&PhmCfgMonitor::status85Cb,
                this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (count >= 25) {
            PHMS_WARN << "PhmCfgMonitor::Init config param '85_status' failed";
        }
        else {
            PHMS_INFO << "PhmCfgMonitor::Init config param '85_status' success";
        }

        res = CONFIG_TIME_OUT;
        count = 0;
        while (CONFIG_OK != res && count++ < 25) {
            PHMS_DEBUG << "PhmCfgMonitor::Init config param 'mode_req' wating......";
            res = ConfigParam::Instance()->MonitorParam<uint8_t>("system/mode_req", std::bind(&PhmCfgMonitor::modeReqCb,
                this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (count >= 25) {
            PHMS_WARN << "PhmCfgMonitor::Init config param 'mode_req' failed";
        }
        else {
            PHMS_INFO << "PhmCfgMonitor::Init config param 'mode_req' success";
        }
    };
    monitor_thread_ = std::thread(f);
}

void
PhmCfgMonitor::DeInit()
{
    PHMS_INFO << "PhmCfgMonitor::DeInit";
    if (time_mgr_ != nullptr) {
        PHMS_INFO << "PhmCfgMonitor::DeInit fd:" << timerFd_;
        time_mgr_->StopFdTimer(timerFd_);
        time_mgr_->DeInit();
    }

    ConfigParam::Instance()->UnMonitorParam("system/power_mode");
    ConfigParam::Instance()->UnMonitorParam("system/calibrate_status");
    ConfigParam::Instance()->UnMonitorParam("system/85_status");
    ConfigParam::Instance()->UnMonitorParam("system/mode_req");
    ConfigParam::Instance()->DeInit();

    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void
PhmCfgMonitor::powerModeCb(const std::string& clientname, const std::string& key, const std::uint8_t& value)
{
    PHMS_INFO << "PhmCfgMonitor::powerModeCb clientname:" << clientname << ",key:" << key << ",value:" << (int32_t)value;
    /*
        0x0 OFF
        0x1 ACC
        0x2 on
        0x3 Crank
    */
    if (0x00 == value) {
        FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_POWERMODE_OFF);
    }
    else {
        FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_NONE);
    }
}

void
PhmCfgMonitor::calibrateStatusCb(const std::string& clientname, const std::string& key, const std::uint8_t& value)
{
    PHMS_INFO << "PhmCfgMonitor::calibrateStatusCb clientname:" << clientname << ",key:" << key << ",value:" << (int32_t)value;
    /*
        0x01    EOL标定开始
        0x02    EOL标定结束
        0x03    行车camera售后静态标定开始
        0x04    行车camera售后静态标定结束
        0x05    行车camera售后动态标定开始
        0x06    行车camera售后动态标定结束
        0x07    泊车camera售后静态标定开始
        0x08    泊车camera售后静态标定结束
        0x09    泊车camera售后动态标定开始
        0x0a    泊车camera售后动态标定结束
        0x0b    Lidar售后静态标定开始
        0x0c    Lidar售后静态标定结束
        0x0d    Lidar售后动态标定开始
        0x0e    Lidar售后动态标定结束
    */
    switch (value) {
    case 0x01:
    case 0x03:
    case 0x05:
    case 0x07:
    case 0x09:
    case 0x0b:
    case 0x0d:
        {
            FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_CALIBRATION);
        }
        break;
    case 0x02:
    case 0x04:
    case 0x06:
    case 0x08:
    case 0x0a:
    case 0x0c:
    case 0x0e:
        {
            FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_NONE);
        }
        break;
    default:
        PHMS_WARN << "PhmCfgMonitor::calibrateStatusCb invalid data";
        break;
    }
}

void
PhmCfgMonitor::status85Cb(const std::string& clientname, const std::string& key, const std::uint8_t& value)
{
    /*
        0x01 85打开
        0x02 85关闭
    */
    PHMS_INFO << "PhmCfgMonitor::status85Cb clientname:" << clientname << ",key:" << key << ",value:" << (int32_t)value;
    if (0x01 == value) {
        FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_85);
    }
    else if (0x02 == value) {
        FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_NONE);
    }
    else {
        PHMS_WARN << "PhmCfgMonitor::status85Cb invalid data";
    }
}

void
PhmCfgMonitor::modeReqCb(const std::string& clientname, const std::string& key, const std::uint8_t& value)
{
    PHMS_INFO << "PhmCfgMonitor::modeReqCb clientname:" << clientname << ",key:" << key << ",value:" << (int32_t)value;
    if (value > 0) {
        FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_RUNNING_MODE);
        time_mgr_->StartFdTimer(timerFd_, INHIBIT_TIME, std::bind(&PhmCfgMonitor::modeReqTimeoutCb, this, std::placeholders::_1), NULL, false);

        uint8_t fm_stat_ready = 1;
        ConfigParam::Instance()->SetParam<std::uint8_t>("system/fm_stat", fm_stat_ready);

        if (2 == m_lastMode && 1 == value) {
            PHMS_INFO << "PhmCfgMonitor::modeReqCb set recover";
            // parking to driving
            bRecoverFault = true;
        }
        else if (1 == m_lastMode && 2 == value) {
            // driving to parking
        }
        else {
        }

        m_lastMode = value;
    }
    else {
        timerFd_ = -1;
        uint8_t fm_stat_notready = 0;
        ConfigParam::Instance()->SetParam<std::uint8_t>("system/fm_stat", fm_stat_notready);
    }
}

void
PhmCfgMonitor::modeReqTimeoutCb(void* data)
{
    PHMS_INFO << "PhmCfgMonitor::modeReqTimeoutCb enter! ";
    timerFd_ = -1;
    FaultDispatcher::getInstance()->SendInhibitType(PHM_INHIBIT_TYPE_NONE);

    if (true == bRecoverFault) {
        PHMS_INFO << "PhmCfgMonitor::modeReqTimeoutCb recover";
        bRecoverFault = false;

        // TODO Recover fault if need
        // std::unordered_map<uint32_t, uint32_t> outEventFaultMap;
        // HzFaultReceiveService::Instance()->GetEventFaultMap(outEventFaultMap);
        // PHMS_INFO << "PhmCfgMonitor::modeReqTimeoutCb recover size:" << outEventFaultMap.size();
        // for (auto ef : outEventFaultMap) {
        //     if ((1 == ef.second) && (8005 == ef.first / 100)) {
        //         PHMS_INFO << "PhmCfgMonitor::modeReqTimeoutCb faultKey:" << ef.first;
        //         HzFaultData fault;
        //         fault.faultId = ef.first / 100;
        //         fault.faultObj = ef.first % 100;
        //         fault.faultStatus = 0;
        //         auto local_time = FaultMgrUtil::GetTimeNow();
        //         fault.faultOccurTime_sec = FaultMgrUtil::GetTimeNowSec(local_time);
        //         fault.faultOccurTime_nsec = FaultMgrUtil::GetTimeNowNanoSec(local_time);
        //         HzFaultReceiveService::Instance()->ReportAlarm(fault, 0);
        //     }
        // }
    }
}


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon