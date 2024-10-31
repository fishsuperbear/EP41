#ifndef _SM_SCHEDULE_CENTER_H_
#define _SM_SCHEDULE_CENTER_H_

#include <unistd.h>
#include <cstdint>
#include <unordered_map>
#include "config_param.h"
#include "adf/include/log.h"

enum RunningMode {
    RUNNING_MODE_ALL = 0,
    RUNNING_MODE_PILOT = 1,
    RUNNING_MODE_PARKING = 2,
    RUNNING_MODE_UNINIT = 0xFF
};

enum ModeReq {
    MODE_REQ_NULL = 0,
    MODE_REQ_PILOT = 1,
    MODE_REQ_PARKING = 2,
};

enum PreModuleStat {
    PRE_MODULE_NOT_READY = 0,
    PRE_MODULE_READY = 1,
};

static uint64_t GetCurrTimeStampMs() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);

    return time.tv_sec * 1000 + time.tv_nsec / 1000 / 1000;
}

class ScheduleCenter {
public:
    int32_t Init() {
        hozon::netaos::cfg::ConfigParam::Instance()->Init();
        NODE_LOG_INFO << "Init HzCfgClient.";

        NODE_LOG_INFO << "Wait for HzCfgClient online.";
        usleep(100*1000);

        // default is pilot mode
        int32_t ret = hozon::netaos::cfg::ConfigParam::Instance()->SetParam<uint8_t>("system/running_mode", static_cast<uint8_t>(RUNNING_MODE_PILOT));
        if (ret != 1) {
            NODE_LOG_ERROR << "Fail to set CfgClient, ret " << ret << ", try later.";
            _cached_running_mode = RUNNING_MODE_UNINIT;
            // just try later 
        }
        else {
            NODE_LOG_INFO << "Init running mode to PILOT";
            _cached_running_mode = RUNNING_MODE_PILOT;
        }

        // monitor mode_req
        _mode_req = MODE_REQ_NULL;
        _mode_req_time_ms = 0;
        NODE_LOG_INFO << "Going to monitor mode_req";
        ret = hozon::netaos::cfg::ConfigParam::Instance()->MonitorParam<uint8_t>("system/mode_req", 
                [this](const std::string& domain, const std::string& key, const uint8_t& value) {
            NODE_LOG_INFO << "Receive mode_req notify, last is " << _mode_req;
            int32_t ret = hozon::netaos::cfg::ConfigParam::Instance()->GetParam<uint8_t>("system/mode_req", _mode_req);
            if (ret != hozon::netaos::cfg::CONFIG_OK) {
                NODE_LOG_WARN << "Fail to get query mode_req " << ret;
                return;
            }
            _mode_req_time_ms = GetCurrTimeStampMs();
            NODE_LOG_INFO << "Receive new mode_req " << _mode_req;
        });
        if (ret != 1) {
            NODE_LOG_ERROR << "Fail to monitor mode_req " << ret;
            return -1;
        }
        NODE_LOG_INFO << "Succ to monitor mode_req";
        
        // monitor fm_stat
        NODE_LOG_INFO << "Going to monitor fm_stat";
        _pre_module_stats["system/fm_stat"] = PRE_MODULE_NOT_READY;
        ret = hozon::netaos::cfg::ConfigParam::Instance()->MonitorParam<uint8_t>("system/fm_stat", 
                std::bind(&ScheduleCenter::ModuleStatChangeCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        if (ret != 1) {
            NODE_LOG_ERROR << "Fail to monitor fm_stat " << ret;
            return -1;
        }
        NODE_LOG_INFO << "Succ to monitor fm_stat";

        return 0;
    }

    void Release() {
        NODE_LOG_INFO << "Going to deinit CfgClient.";
        hozon::netaos::cfg::ConfigParam::Instance()->DeInit();
        NODE_LOG_INFO << "Deinit CfgClient end.";
    }

private:
    void ModuleStatChangeCallback(const std::string& domain, const std::string& key, const uint8_t& value) {
        NODE_LOG_INFO << "Receive " << key << " notify";
        uint8_t stat = 0;
        int32_t ret = hozon::netaos::cfg::ConfigParam::Instance()->GetParam<uint8_t>(key, stat);
        if (ret != hozon::netaos::cfg::CONFIG_OK) {
            NODE_LOG_WARN << "Fail to get query " << key << " " << ret;
            return;
        }
        NODE_LOG_INFO << "Receive new " << key << " " << (int32_t)stat;
        _pre_module_stats[key] = stat;

        if (_mode_req == MODE_REQ_NULL) {
            NODE_LOG_INFO << "No mode_req, ignore it.";
        } else if (_mode_req == MODE_REQ_PILOT) {
            if (CheckPreModuleReady()) {
                ++_count;
                NODE_LOG_INFO << "Switch to PILOT_MODE, time since last mode_req " << GetCurrTimeStampMs() - _mode_req_time_ms << "(ms), count: " << _count;
                SetRunningMode(static_cast<uint8_t>(RUNNING_MODE_PILOT));
                SetModeReq(static_cast<uint8_t>(MODE_REQ_NULL));
            }
        } else if (_mode_req == MODE_REQ_PARKING) {
            if (CheckPreModuleReady()) {
                ++_count;
                NODE_LOG_INFO << "Switch to PARKING_MODE, time since last mode_req " << GetCurrTimeStampMs() - _mode_req_time_ms << "(ms), count: " << _count;
                SetRunningMode(static_cast<uint8_t>(RUNNING_MODE_PARKING));
                SetModeReq(static_cast<uint8_t>(MODE_REQ_NULL));
            }
        }
    }

    bool CheckPreModuleReady() {
        bool ready = true;
        for (const auto& ele : _pre_module_stats) {
            if (ele.second != PRE_MODULE_READY) {
                NODE_LOG_DEBUG << ele.first << " is not ready.";
                ready = false;
            }
        }

        return ready;
    }

    void SetRunningMode(uint8_t running_mode) {
        NODE_LOG_INFO << "Switch running_mode to " << (int32_t)running_mode;
        int32_t ret = hozon::netaos::cfg::ConfigParam::Instance()->SetParam<uint8_t>("system/running_mode", running_mode);
        if (ret != 1) {
            NODE_LOG_ERROR << "Fail to set running_mode, ret " << ret;
            return;
        }
    }

    void SetModeReq(uint8_t mode_req) {
        NODE_LOG_INFO << "Set mode_req to " << (int32_t)mode_req;
        int32_t ret = hozon::netaos::cfg::ConfigParam::Instance()->SetParam<uint8_t>("system/mode_req", mode_req);
        if (ret != 1) {
            NODE_LOG_ERROR << "Fail to set mode_req, ret " << ret;
            return;
        }
    }

private:
    uint8_t _cached_running_mode;
    uint8_t _mode_req;
    uint64_t _mode_req_time_ms;
    uint64_t _count = 0;
    std::unordered_map<std::string, uint8_t> _pre_module_stats;
};

#endif