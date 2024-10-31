#include "sys_statemgr/include/power_manager_svr.h"
#include "sys_statemgr/include/state_manager.h"
#include "sys_statemgr/include/logger.h"

namespace hozon {
namespace netaos {
namespace ssm {


PowerManagerServiceSvr::PowerManagerServiceSvr(std::string instance): SocPowerServiceSkeleton::SocPowerServiceSkeleton(ara::com::InstanceIdentifier(instance.c_str())) {
    m_stopFlag = 0;
    m_sys_state = 0;
    m_sm_state = 0;
    SSM_LOG_INFO << "create instance identifier:" << instance;
}

PowerManagerServiceSvr::~PowerManagerServiceSvr() {
}

void PowerManagerServiceSvr::DeInit() {
    SSM_LOG_INFO << __func__;
    m_stopFlag = 1;
    StopOfferService();
    if (m_thr_pms.joinable()) {
        m_thr_pms.join();
    }
}

void PowerManagerServiceSvr::Init(std::shared_ptr<StateManager> ptr) {
    SSM_LOG_INFO << __func__;
    m_smgr = ptr;
    OfferService();
}

void PowerManagerServiceSvr::Run() {
    TriggerSysStateEvent();
}

void PowerManagerServiceSvr::TriggerSysStateEvent() {
    m_thr_pms = std::thread([this]() {
    while (!m_stopFlag) {
        uint8_t s_state = m_smgr->GetSysState();
        if (s_state != 0) {
            ara::com::SampleAllocateePtr<::hozon::netaos::SocSysState> sstate = SocSystemState.Allocate();
            sstate->sys_state = s_state;
            sstate->sm_state = m_smgr->GetSMState();
            if (m_sys_state != sstate->sys_state || m_sm_state != sstate->sm_state) {
                m_sys_state = sstate->sys_state;
                m_sm_state = sstate->sm_state;
                SSM_LOG_INFO << ">>> soc state changed,systate:" << static_cast<uint8_t>(sstate->sys_state) << ",smstate:" << static_cast<uint8_t>(sstate->sm_state);
            }
            SocSystemState.Send(std::move(sstate));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    });
}

}}}
