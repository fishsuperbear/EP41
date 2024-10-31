#include "sys_statemgr/include/mcustate_service_cli.h"
#include "sys_statemgr/include/phm_client_instance.h"
#include "sys_statemgr/include/state_manager.h"
#include "sys_statemgr/include/logger.h"

namespace hozon {
namespace netaos {
namespace ssm {


McuStateServiceCli::McuStateServiceCli(std::string instance) : m_instance_id(instance), m_proxy(nullptr) {
    m_l2_state = 0;
    m_l3_state = 0;
}

McuStateServiceCli::~McuStateServiceCli() {
}

void McuStateServiceCli::Init(std::shared_ptr<StateManager> ptr) {
    SSM_LOG_INFO << __func__;
    m_smgr = ptr;
    StartFindService();
}

void McuStateServiceCli::DeInit() {
    SSM_LOG_INFO << __func__;
    StopFindService();
}

void McuStateServiceCli::Run() {
    // std::this_thread::sleep_for(std::chrono::milliseconds(40000));
    // SSM_LOG_INFO << "=== McuStateServiceCli Req SoC Restart ===";
    // hozon::netaos::PowerModeEnum powermode = hozon::netaos::PowerModeEnum::Restart;
    // SocPowerModeRequest(powermode);
}

void McuStateServiceCli::StartFindService() {
    SSM_LOG_INFO << "start find service";

    m_handle = hozon::netaos::v1::proxy::McuStateServiceProxy::StartFindService(
        [this]( ara::com::ServiceHandleContainer<ara::com::HandleType> handles, ara::com::FindServiceHandle handler ) {
            (void) handler;
                SSM_LOG_INFO << "service found size:" << static_cast<uint16_t>( (handles.size()) );
                McuStateServiceCli::ServiceAvailabilityCallback( std::move(handles) );
        }, ara::com::InstanceIdentifier("1")
    );

    SSM_LOG_INFO << "find service end";
}

void McuStateServiceCli::StopFindService() {
    if (m_proxy) {
        m_proxy->McuSystemState.Unsubscribe();
        m_proxy->McuSystemState.UnsetReceiveHandler();
        m_proxy->StopFindService(m_handle);
    }
}

void McuStateServiceCli::ServiceAvailabilityCallback( ara::com::ServiceHandleContainer<ara::com::HandleType> handles ) {
    SSM_LOG_INFO << "service availability callback";
    if ( handles.size() > 0U ) {
        const size_t maxCount = 10U;
        for (auto it : handles) {
            std::lock_guard<std::mutex> lock(m_proxy_mtx);
            if ( m_proxy == nullptr ) {
                m_proxy = std::make_shared<hozon::netaos::v1::proxy::McuStateServiceProxy>(it);
                m_proxy->McuSystemState.SetReceiveHandler([this]() {McuStateServiceCli::OnRecvSysStateCallback(); });
                SSM_LOG_INFO << "created proxy succ with handle instance 1";
            }
        }
        if ( m_proxy == nullptr ) {
            SSM_LOG_ERROR << "create Proxy failed";
        } else { 
            m_proxy->McuSystemState.Subscribe(maxCount);
        }
    } else {
        SSM_LOG_ERROR << "service disconnected";
        m_proxy = nullptr;
    }
}


void McuStateServiceCli::OnRecvSysStateCallback() {
    if (m_proxy == nullptr) {
        SSM_LOG_WARN << "proxy is null";
        return;
    }
    m_proxy->McuSystemState.GetNewSamples([this](ara::com::SamplePtr<::hozon::netaos::McuSysState const> ptr) {
        if (m_l2_state != ptr->L2_state || m_l3_state != ptr->L3_state) {
            m_l2_state = ptr->L2_state;
            m_l3_state = ptr->L3_state;
            m_smgr->SetMcuL2State(m_l2_state);
            m_smgr->SetMcuL3State(m_l3_state);
            SSM_LOG_INFO << "<<< recv mcu state changed, L2_state:" << static_cast<uint8_t>(ptr->L2_state) << ",L3_state:" << static_cast<uint8_t>(ptr->L3_state);
        }
    });
}

int32_t McuStateServiceCli::SocPowerModeRequest(hozon::netaos::PowerModeEnum& powermode) {
    // std::thread([this](hozon::netaos::PowerModeEnum pmode) {
    int32_t res = -1;
    if (m_proxy != nullptr) {
        SSM_LOG_INFO << "==>> req change mode:" << static_cast<uint8_t>(powermode);
        auto future = m_proxy->PowerModeReq(powermode);
        auto state_RequestResult = future.wait_for(std::chrono::milliseconds(3000));
        if(ara::core::future_status::ready == state_RequestResult) {
            auto result = future.GetResult();
            if (result.HasValue()) {
                auto output = result.Value();
                SSM_LOG_INFO << "< soc power request result:" << static_cast<uint8_t>(output.RequestResult) << " >";
                res = 0;
            } else {
                // auto error = result.Error();
                SSM_LOG_ERROR << "< recv the error code >";
            }
        } else {
            SSM_LOG_ERROR << "< req wait result <timeout> >";
            //fault report once
            uint32_t faultId = 4320;
            uint8_t faultObj = 1;
            uint8_t faultStatus = 1;
            SendFault_t sendOnceFault(faultId, faultObj, faultStatus);
            PhmClientInstance::getInstance()->ReportFault(sendOnceFault);
        }
    } else {
        SSM_LOG_ERROR << "mcu state service proxy is null";
    }
    // },powermode).detach();
    return res;
}

}}}
