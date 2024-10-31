#include <sys/prctl.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>
#include "sm/include/state_client.h"
#include "sys_statemgr/include/logger.h"
#include "sys_statemgr/include/sys_define.h"
#include "sys_statemgr/include/state_manager.h"
#include "sys_statemgr/include/phm_client_instance.h"

namespace hozon {
namespace netaos {
namespace ssm {

const std::string soc_sys_state[5] = {"Startup", "Factory", "Normal", "Update", "Standby"};
const std::string mcu_sys_state[6] = {"Default", "Startup", "Factory", "Normal", "Update", "PreStandAlone"};

StateManager::StateManager() {
    m_sys_state = 0;  /* default */
    m_mach_state = 1; /* driving */
    m_l2_state = 0;
    m_l3_state = 0;
    m_stopFlag = 0;
}

StateManager::~StateManager() {}

void StateManager::DeInit() {
    SSM_LOG_INFO << __func__;
    m_stopFlag = 1;
    if (m_stat_thr.joinable()) {
        m_stat_thr.join();
    }
    if (m_sync_thr.joinable()) {
        m_sync_thr.join();
    }
    ConfigParam::Instance()->UnMonitorParam("system/running_mode");
    ConfigParam::Instance()->DeInit();
}

int32_t StateManager::Init() {
    SSM_LOG_INFO << __func__;
    m_sm_cli = std::make_shared<StateClientZmq>();
    CfgResultCode res = ConfigParam::Instance()->Init(3000);
    if (res == 0) {
        SSM_LOG_ERROR << "configparam init error:";
        return -1;
    }
    std::string def_state = "Startup";
    ConfigParam::Instance()->SetDefaultParam<std::string>("system/soc_status", def_state);
    ConfigParam::Instance()->SetDefaultParam<std::string>("system/mcu_status", def_state);
    ConfigParam::Instance()->MonitorParam<uint8_t>("system/running_mode", std::bind(&StateManager::OnRecvSMChangedCallback,
        this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    ConfigParam::Instance()->MonitorParam<std::string>("dids/2910", std::bind(&StateManager::OnRecvModeChangedCallback,
        this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    return 0;
}

void StateManager::Run() {
    m_stat_thr = std::thread([this]() {
        std::string curmode = "";
        StateClient smcli;
        uint8_t sstate = 0;
        while (!m_stopFlag) {
            int32_t ret = smcli.GetCurrMode(curmode);
            if (0 == ret) {
                if (curmode == "Normal") {
                    sstate = 0x02;
                } else if (curmode == "Update") {
                    sstate = 0x03;
                } else if (curmode == "Standby") {
                    sstate = 0x04;
                } else if (curmode == "Factory") {
                    sstate = 0x01;
                }
                SetSysState(sstate, curmode);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    this->SysStateSyncMonitor();
}

int32_t StateManager::SwitchMode(const std::string& mode) {
    SSM_LOG_INFO << "switch mode:" <<mode;
    int32_t res = 0;
    for (size_t i = 0; i < 3; i++) {
        if (i > 0) { 
            SSM_LOG_WARN << "retry "<<i+1;
            uint32_t faultId = 4310;
            uint8_t faultObj = 1;
            uint8_t faultStatus = 1;
            SendFault_t sendOnceFault(faultId, faultObj, faultStatus);
            PhmClientInstance::getInstance()->ReportFault(sendOnceFault);
        }
        res = m_sm_cli->SwitchMode(mode);
        switch (res)
        {
            case 0:
                SSM_LOG_INFO << "succ";
                return res;
            case 1:
                SSM_LOG_ERROR << "fail";
                return res;
            case 2:
            case 3:
                SSM_LOG_WARN << "reject";
                std::this_thread::sleep_for(std::chrono::seconds(2u));
                break;
            default:
                break;
        }
    }
    if (0 == res) {
        //fault report once
        uint32_t faultId = 4310;
        uint8_t faultObj = 1;
        uint8_t faultStatus = 0;
        SendFault_t sendOnceFault(faultId, faultObj, faultStatus);
        PhmClientInstance::getInstance()->ReportFault(sendOnceFault);
    }

    return res;
}

int32_t StateManager::StopMode() {
    return m_sm_cli->StopMode();
}

void StateManager::OnRecvSMChangedCallback(const std::string& domain, const std::string& key, const uint8_t& value) {
    SSM_LOG_INFO << "recv l3 state machine changed:"<< value;
    this->SetSMState(value);
}

void StateManager::OnRecvModeChangedCallback(const std::string& domain, const std::string& key, const std::string& value) {
    SSM_LOG_INFO << "recv eol mode changed:"<< value << "(cur mode:" << m_sys_state <<")";
    int32_t ret = 0;
    /* Factory Mode */
    if(value == "01") {
        ret = m_sm_cli->SetDefaultMode("Factory");
        this->SwitchMode("Factory");
    } else if(value == "02") {
        /* Normal Mode */
        ret = m_sm_cli->SetDefaultMode("Normal");
        this->SwitchMode("Normal");
    }
    if (ret == 0) {
        SSM_LOG_ERROR << "set sys default mode succ";
    } else {
        SSM_LOG_ERROR << "set sys default mode fail";
    }
}

uint8_t StateManager::GetSysState() {
    std::shared_lock<std::shared_timed_mutex> rLock(m_mutex_sstate);
    return m_sys_state;
}

uint8_t StateManager::GetSMState() {
    std::shared_lock<std::shared_timed_mutex> rLock(m_mutex_mstate);
    return m_mach_state;
}

void StateManager::SetSysState(uint8_t state, const std::string &mode) {
    std::lock_guard<std::shared_timed_mutex> lock(m_mutex_sstate);
    if (m_sys_state != state) {
        m_sys_state = state;
        ConfigParam::Instance()->SetParam<std::string>("system/soc_status", mode, ConfigPersistType::CONFIG_SYNC_PERSIST);
    }
}

void StateManager::SetSMState(uint8_t state) {
    std::lock_guard<std::shared_timed_mutex> lock(m_mutex_mstate);
    if (m_mach_state != state) {
        m_mach_state = state;
    }
}

uint8_t StateManager::GetMcuL2State() {
    std::shared_lock<std::shared_timed_mutex> rLock(m_mutex_l2state);
    return m_l2_state;
}

uint8_t StateManager::GetMcuL3tate() {
    std::shared_lock<std::shared_timed_mutex> rLock(m_mutex_l3state);
    return m_l3_state;
}

void StateManager::SetMcuL2State(uint8_t l2state) {
    std::lock_guard<std::shared_timed_mutex> lock(m_mutex_l2state);
    if (m_l2_state != l2state) {
        m_l2_state = l2state;
        std::string mode = mcu_sys_state[l2state];
        ConfigParam::Instance()->SetParam<std::string>("system/mcu_status", mode, ConfigPersistType::CONFIG_SYNC_PERSIST);
    }
}

void StateManager::SetMcuL3State(uint8_t l3state) {
    std::lock_guard<std::shared_timed_mutex> lock(m_mutex_l3state);
    if (m_l3_state != l3state) {
        m_l3_state = l3state;
    }
}

void StateManager::SysStateSyncMonitor() {
    static int32_t count = 20;
    m_sync_thr = std::thread([this]() {
        while (!m_stopFlag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            uint8_t sstate = GetSysState();
            uint8_t l2stat = GetMcuL2State();
            if(sstate +1 == l2stat) {
                count = 20;
            } else if ((sstate >= 0) && (sstate <= 3) && (l2stat != sstate + 1)) {
                if (--count <= 0) {
                    count = 20;
                    //fault report once
                    uint32_t faultId = 4320;
                    uint8_t faultObj = 2;
                    uint8_t faultStatus = 1;
                    SendFault_t sendOnceFault(faultId, faultObj, faultStatus);
                    PhmClientInstance::getInstance()->ReportFault(sendOnceFault);
                    SSM_LOG_ERROR << "sys state sync failure: sstate " << sstate << " l2state " << l2stat;
                }
            }
        }
    });
}

}}}
