#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <bits/types/FILE.h>
#include <thread>
#include <iostream>
#include <unistd.h>
#include "sensor_manager.h"
#include "logger.h"
#include "param_config.h"
#include "sensor_proxy.h"
#include "someip_skeleton.h"


namespace hozon {
namespace netaos {
namespace sensor {
// SensorManager::SensorManager() {}
template <typename T>
std::shared_ptr<hozon::netaos::sensor::SensorProxyBase> CreateProxy() {
    return std::make_shared<T>();
}

std::unordered_map<std::string, std::function<std::shared_ptr<SensorProxyBase>()>> g_someip_map = {
    {"mcu_to_soc", CreateProxy<SensorProxy>}

};

template <typename T>
std::shared_ptr<hozon::netaos::sensor::SomeipSkeleton> CreateSkeleton() {
    return std::make_shared<T>();
}
std::unordered_map<std::string, std::function<std::shared_ptr<SomeipSkeleton>()>> g_skeleton_someip_map = {
    {"someip_skeleton", CreateSkeleton<SomeipSkeleton>}
};

cm_info_map g_cm_info_map[] = {
    // name  domainid    topic name
    {"imuins", 0, "/soc/imuinsinfo"},
    {"chassis", 0, "/soc/chassis"},
    {"gnss", 0, "/soc/gnssinfo"},
    {"mcu2ego", 0, "/soc/mcu2ego"},
    {"pnc", 0, "/soc/statemachine"},
    {"uss", 0, "/soc/ussrawdata"},

    {"radarfront", 0, "/soc/radarfront"},
    {"radarcorner1", 0, "/soc/radarcorner1"}, // FR
    {"radarcorner2", 0, "/soc/radarcorner2"}, // FL
    {"radarcorner3", 0, "/soc/radarcorner3"}, // RR
    {"radarcorner4", 0, "/soc/radarcorner4"}, // RL
};

struct {
    std::string name;
    uint32_t is_nnp;
} g_cm_proxy_map[] = {
    {"ego2mcu_chassis",     0},
    {"apa2mcu_chassis",     0},
    {"nnplane",            1},
    {"hpplane",            2},
    {"nnplocation",        1},
    {"hpplocation",        2},
    {"nnpobject",          1},
    {"hppobject",          2},
    {"sm2mcu",              0},
    {"parkinglot2hmi_2",    0},
    {"ihbc",            0},
    {"guard_mode",         0},
    {"mod",         0},
    {"tsrtlr",         0},
};

int SensorManager::Init(std::string &config, uint32_t is_nnp) {

    SENSOR_LOG_INFO << "Creat cm proxy instance.";
    // start internal report EM
    _cm_proxy = std::make_shared<CmProxy>();
    if(_cm_proxy != nullptr) {
        for(auto cm_proxy_p : g_cm_proxy_map) {
            // if(cm_proxy_p.is_nnp == is_nnp || cm_proxy_p.is_nnp == 0) {
            _cm_proxy->Register(cm_proxy_p.name, std::bind(&SensorManager::WriteSomeip,
                this, std::placeholders::_1, cm_proxy_p.name));
            // }
        }
        _cm_proxy->Start(config);
    }

    // DDS skeleoton 初始化
    for(auto cm_info : g_cm_info_map) {
        _skeleton_instance_map[cm_info.name] = std::make_shared<Skeleton>(cm_info.domainID, cm_info.topic);
    }

    // init param config will block 3 seconds;
    ParamConfig::GetInstance().Init();

    // Someip接收, need get config pararmeter from param config.
    for(auto someip_p : g_someip_map) {
        _proxy_instance_map[someip_p.first] = someip_p.second();  // 批量调用CreateProxy

        _proxy_instance_map[someip_p.first]->RegisgerWriteFunc(std::bind(&SensorManager::Write,
             this, std::placeholders::_1, std::placeholders::_2));// 数据接收后 回调，发送到DDS

        _proxy_instance_map[someip_p.first]->Init();
    }

    // someip send
    SENSOR_LOG_INFO << "Creat someip skeleton instance.";
    InitRunningMode(is_nnp);
    for(auto someip_skeleton_p : g_skeleton_someip_map) {
        _someip_skeleton_map[someip_skeleton_p.first] = someip_skeleton_p.second();
        _someip_skeleton_map[someip_skeleton_p.first]->Init(is_nnp);
    }

    // chassis method to ota
    _chassis_server = std::make_shared<ChassisServer>();
    _chassis_server->Start(0, "/soc/chassis_ota_method");

    SENSOR_LOG_INFO << "sensor manager init successful.";
    return 0;
}
int SensorManager::InitRunningMode(uint32_t is_nnp) {
    uint8_t running_mode = 0;
    if( ParamConfig::GetInstance().GetParam("system/running_mode", running_mode) != 0
        || (running_mode != 1 && running_mode != 2)) {
        running_mode = is_nnp;
        SENSOR_LOG_INFO << "Config get param system/running_mode init fail, use yaml config running mode: "
                 << is_nnp;
    }
    else {
        SENSOR_LOG_INFO << "Config get param system/running_mode init ok, running mode: " << running_mode;
    }
    ChangeRunningMode(true, running_mode);
    ParamConfig::GetInstance().SetMonitor("system/running_mode",
    [this](const std::string& client, const std::string& key, const uint8_t& value) {
            SENSOR_LOG_INFO << "Config param " << client << ": " << key << ": " << value;
            if (key == "system/running_mode" && (value == 1 || value == 2)) {
                // std::lock_guard<std::mutex> lck(_running_mode_mutex);
                ChangeRunningMode(false, value);
            }
        });
    return 0;
}

int SensorManager::ChangeRunningMode(bool is_init, uint8_t running_mode) {
    if(_cm_proxy != nullptr) {
        for(auto cm_proxy_p : g_cm_proxy_map) {
            if(cm_proxy_p.is_nnp != 0 && cm_proxy_p.is_nnp != running_mode) {
                _cm_proxy->Pause(cm_proxy_p.name);
                SENSOR_LOG_INFO << "Pause : " << cm_proxy_p.name;
            }
            else if(is_init == false && cm_proxy_p.is_nnp == running_mode) {
                _cm_proxy->Resume(cm_proxy_p.name);
                SENSOR_LOG_INFO << "Resume : " << cm_proxy_p.name;
            }
        }
    }
    return 0;
}
int32_t SensorManager::WriteSomeip(adf::NodeBundle* input, std::string name) {
    // std::lock_guard<std::recursive_mutex> lck(_someip_skeleton_map_mt);
    SENSOR_LOG_DEBUG << "WriteSomeip data: " << name << " start.";
    if(_someip_skeleton_map.find("someip_skeleton") != _someip_skeleton_map.end()) {
        _someip_skeleton_map["someip_skeleton"]->Write(name, *input);
    }
    return 0;
}

int SensorManager::Write(std::string name, std::shared_ptr<void> data) {
    // SENSOR_LOG_INFO << name << " start data send.";
    // std::lock_guard<std::recursive_mutex> lck(_skeleont_instance_map_mt);    // 递归锁，同一线程不会出现死锁
    if(_skeleton_instance_map.find(name) == _skeleton_instance_map.end()) {
        SENSOR_LOG_ERROR << "Fail find " << name << " in skeleton instance.";
        return -1;
    }
    // SENSOR_LOG_INFO << name << " data send.";
    if(_skeleton_instance_map[name]->Write(data)) {
        SENSOR_LOG_ERROR << name << " data send fail.";
    }
    else {
        SENSOR_LOG_DEBUG << name << " data send successful.";
    }
    return 0;
}

int SensorManager::Stop() {
    SENSOR_LOG_INFO << "Sensormanager stop...";

    _chassis_server->Stop();

    for(auto instance : _proxy_instance_map) {
        instance.second->Deinit();
    }
    for(auto instance : _skeleton_instance_map) {
        instance.second->Deinit();
    }
    // for(auto instance : _cm_proxy_map) {
    //     instance.second->Deinit();
    // }

    if (_cm_proxy != nullptr) {
        _cm_proxy->Deinit();
    }

    for(auto instance : _someip_skeleton_map) {
        instance.second->Deinit();
    }

    ParamConfig::GetInstance().DeInit();
    SENSOR_LOG_INFO << "Sensormanager stop successful.";
    return 0;
}

int SensorManager::Run() {
    // for(auto instance : _cm_proxy_map) {
    //     SENSOR_LOG_INFO << "Run send data " << instance.first;
    //     instance.second->Run();
    // }
    return 0;
}
void SensorManager::WaitStop(void) {
    SENSOR_LOG_INFO << "Sensormanager WaitStop.";
    if(_cm_proxy != nullptr) {
        _cm_proxy->WaitStop();
    }
    SENSOR_LOG_INFO << "Sensormanager WaitStop successful.";
}

}
}
}
