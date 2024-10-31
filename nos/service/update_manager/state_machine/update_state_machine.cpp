#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/state_machine/state_file_manager.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/manager/uds_command_controller.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "update_manager/config/sensor_entity_manager.h"
#include "update_manager/update_check/update_check.h"

namespace hozon {
namespace netaos {
namespace update {

UpdateStateMachine* UpdateStateMachine::m_pInstance = nullptr;
std::mutex UpdateStateMachine::m_mtx;


UpdateStateMachine::UpdateStateMachine()
{
}

UpdateStateMachine::~UpdateStateMachine()
{
}

UpdateStateMachine*
UpdateStateMachine::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new UpdateStateMachine();
        }
    }

    return m_pInstance;
}

void
UpdateStateMachine::InitStateMap()
{
	UM_INFO << "UpdateStateMachine::InitStateMap.";
	AddAllowedTransition(State::NORMAL_IDLE, State::OTA_PRE_UPDATE);
	AddAllowedTransition(State::OTA_PRE_UPDATE, State::OTA_UPDATING);
    AddAllowedTransition(State::OTA_UPDATING, State::OTA_UPDATED);
    AddAllowedTransition(State::OTA_UPDATED, State::OTA_ACTIVING);
    AddAllowedTransition(State::OTA_ACTIVING, State::OTA_ACTIVED);
	AddAllowedTransition(State::OTA_PRE_UPDATE, State::OTA_UPDATE_FAILED);
    AddAllowedTransition(State::OTA_UPDATING, State::OTA_UPDATE_FAILED);
    AddAllowedTransition(State::OTA_ACTIVING, State::OTA_UPDATE_FAILED);
    AddAllowedTransition(State::OTA_ACTIVED, State::NORMAL_IDLE);
    AddAllowedTransition(State::OTA_UPDATE_FAILED, State::NORMAL_IDLE);
	StateFileManager::Instance()->CreateStateFile();
    UM_INFO << "UpdateStateMachine::InitStateMap Done.";
}

void
UpdateStateMachine::Deinit()
{
	UM_INFO << "UpdateStateMachine::Deinit.";
    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "UpdateStateMachine::Deinit Done.";
}

bool 
UpdateStateMachine::SwitchState(State newState, const FailedCode& code)
{
    UPDATE_LOG_D("switchState");
	if (allowedTransitions[currentState].find(newState) != allowedTransitions[currentState].end()) {
    	UPDATE_LOG_D("switchState success, from %s to %s", GetStateString(currentState).c_str(), GetStateString(newState).c_str());
		preState = currentState;
		currentState = newState;
		StateFileManager::Instance()->UpdateStateFile(GetStateString(currentState));
		if (GetStateString(newState) == "OTA_UPDATE_FAILED") {
			SetFailedCode(code);
			UPDATE_LOG_D("UpdateManager switch into Normal Mode.");
        	UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_NORMAL);
        	SwitchState(State::NORMAL_IDLE);
        	PostUpdateProcess();
		} else if (GetStateString(newState) == "OTA_ACTIVED") {
			UPDATE_LOG_D("UpdateManager switch into Normal Mode.");
        	UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_NORMAL);
        	SwitchState(State::NORMAL_IDLE);
        	PostUpdateProcess();
		} else {
			// go on
		}
	} else {
		UPDATE_LOG_E("Invalid transition from %s to %s",GetStateString(currentState).c_str(), GetStateString(newState).c_str());
		return false;
	}
	return true;
}

std::string 
UpdateStateMachine::GetCurrentState() const
{
	return GetStateString(currentState);
}

std::string 
UpdateStateMachine::GetPreState() const
{
	return GetStateString(preState);
}

void 
UpdateStateMachine::SetInitialState(State initialState)
{
	currentState = initialState;
	if (initialState == State::NORMAL_IDLE)
	{
		return;
	}
	StateFileManager::Instance()->UpdateStateFile(GetStateString(currentState));
}

void 
UpdateStateMachine::ForceSetState(State state)
{
	currentState = state;
	StateFileManager::Instance()->UpdateStateFile(GetStateString(currentState));
}


void 
UpdateStateMachine::AddAllowedTransition(State fromState, State toState)
{
	allowedTransitions[fromState].insert(toState);
}
std::string 
UpdateStateMachine::GetStateString(State state) const
{
	std::string stateMsg{};
	switch (state)
	{
	case State::NORMAL_IDLE:
		stateMsg = "NORMAL_IDLE";
		break;
	case State::OTA_PRE_UPDATE:
		stateMsg = "OTA_PRE_UPDATE";
		break;
	case State::OTA_UPDATING:
		stateMsg = "OTA_UPDATING";
		break;
	case State::OTA_UPDATED:
		stateMsg = "OTA_UPDATED";
		break;
	case State::OTA_ACTIVING:
		stateMsg = "OTA_ACTIVING";
		break;
	case State::OTA_ACTIVED:
		stateMsg = "OTA_ACTIVED";
		break;
	case State::OTA_UPDATE_FAILED:
		stateMsg = "OTA_UPDATE_FAILED";
		break;
	
	default:
		stateMsg = "";
		break;
	}
	return stateMsg;
}

uint16_t 
UpdateStateMachine::GetFailedCode()
{
	return static_cast<uint16_t>(errorCode_);
}

std::string 
UpdateStateMachine::GetFailedCodeMsg()
{
	std::string errCodeMsg{};
	switch (errorCode_)
	{
	case FailedCode::PARSE_CONFIG_FAILED:
		errCodeMsg = "PARSE_CONFIG_FAILED";
		break;
	case FailedCode::VERIFY_FAILED:
		errCodeMsg = "VERIFY_FAILED";
		break;
	case FailedCode::CONFIG_DELETED:
		errCodeMsg = "CONFIG_DELETED";
		break;
	case FailedCode::ECU_MODE_INVALIED:
		errCodeMsg = "ECU_MODE_INVALIED";
		break;
	case FailedCode::SENSOR_UPDATE_FAILED:
		errCodeMsg = "SENSOR_UPDATE_FAILED";
		break;
	case FailedCode::SENSOR_NOT_EXIST:
		errCodeMsg = "SENSOR_NOT_EXIST";
		break;
	case FailedCode::SOC_UPDATE_FAILED:
		errCodeMsg = "SOC_UPDATE_FAILED";
		break;
	case FailedCode::MCU_UPDATE_FAILED:
		errCodeMsg = "MCU_UPDATE_FAILED";
		break;
	default:
		errCodeMsg = "DEFAULT";
		break;
	}
	return errCodeMsg;
}

void
UpdateStateMachine::SetFailedCode(const FailedCode& code)
{
	errorCode_ = code;
}

void 
UpdateStateMachine::PostUpdateProcess()
{
    UPDATE_LOG_D("UpdateStateMachine::PostUpdateProcess");
	ConfigManager::Instance().UmountSensors();
    UdsCommandController::Instance()->ResetProcess();
	UdsCommandController::Instance()->SetVersionSameFlag(false);
	DiagAgent::Instance()->ResetTotalProgress();
	SensorEntityManager::Instance()->ResetSensorEntityInfo();
	ConfigManager::Instance().ClearSocUpdate();
	ConfigManager::Instance().ClearSensorUpdate();
	PathClear(UpdateSettings::Instance().PathForUpgrade());
	if (!CmdUpgradeManager::Instance()->IsCmdTriggerUpgrade())
	{
		std::string packageName{};
		auto res = OTAStore::Instance()->ReadPackageNameData(packageName);
		if (res) {
			PathRemove(packageName);
		}
	}
	CmdUpgradeManager::Instance()->SetCmdTriggerUpgradeFlag(false);
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
