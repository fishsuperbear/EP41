#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include "cfg/include/cfg_data_def.h"
#include "cfg/include/config_param.h"
#include "logger.h"

namespace hozon {
namespace netaos {
namespace sensor {
class ParamConfig {
public:
    ParamConfig() 
    : _init_ok(false) {
        _cfg_instance = nullptr;
    }
    static ParamConfig& GetInstance() {
       static ParamConfig instance;
       return instance;
    }
    int InitParamInstance() {
        cfg::CfgResultCode result = _cfg_instance->Init(3000);
        if(result != 1u) {
            SENSOR_LOG_WARN << "Config param init fail." << result;
            return -1;
        }
        else {
            _init_ok = true;
            SENSOR_LOG_INFO << "Config param init ok.";
        }
        return 0;
    }
    int Init() {
        _cfg_instance = cfg::ConfigParam::Instance();
        if(_cfg_instance == nullptr) {
            return -1;
        }
        else {
            InitParamInstance();
        }
        return 0;
    }
    int GetParam(const std::string& key, uint8_t& value) {
        if (_cfg_instance != nullptr) {
            if (_init_ok) {
                cfg::CfgResultCode result = _cfg_instance->GetParam<uint8_t>(key, value);
                if(result == 1u) {
                    SENSOR_LOG_INFO << "GetParam " << key << " success, value: " << value;
                    return 0;
                }
                else {
                    SENSOR_LOG_INFO << "GetParam " << key << " error: " << result;
                }
            }
        }
        return -1;
    }   
    
    int SetMonitor(const std::string& key, 
            const std::function<void(const std::string&, const std::string&, const uint8_t&)> func) {
        if (_cfg_instance != nullptr) {
            if (_init_ok) {
                cfg::CfgResultCode result = _cfg_instance->MonitorParam<uint8_t>(key, func);
                if(result == 1u) {
                    SENSOR_LOG_INFO << "SetMonitor " << key << " success";
                    return 0;
                }
                else {
                    SENSOR_LOG_INFO << "SetMonitor " << key << " error: " << result;
                }
            }
        }
        return -1;
    }
    int DeInit() {
        if (_cfg_instance != nullptr) {
            if(_init_ok) {
            _cfg_instance->UnMonitorParam("hozon/running_mode");
        }
            _cfg_instance->DeInit();
        }
        return 0;
    }
    int SetParam(const std::string& key, uint8_t value) {
        if (_cfg_instance != nullptr) { 
            if (_init_ok) {
                cfg::CfgResultCode result = 
                    _cfg_instance->SetParam<uint8_t>(key, value, cfg::CONFIG_NO_PERSIST);
                if (1 == static_cast<int>(result)) {
                    SENSOR_LOG_INFO << "SetParam " << key << " : " << value   << " success";
                    return 0;
                } else {
                    SENSOR_LOG_WARN << "SetParam " << key << " : " << value << " error: " << result;
                } 
            }
        }
        return -1;
    }

    ~ParamConfig() = default;
    
    
    
private:
    cfg::ConfigParam* _cfg_instance;
    bool _init_ok = false;
};
}
}
}