#include "update_manager/manager/update_manager.h"
#include "update_manager/manager/uds_command_controller.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/agent/download_agent.h"
#include "update_manager/agent/update_agent.h"
#include "update_manager/update_check/update_check.h"
#include "update_manager/config/update_settings.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "update_manager/state_machine/state_file_manager.h"
#include "update_manager/cmd_line_upgrade/sensors_uds_manager.h"
#include "update_manager/record/ota_result_store.h"
#include "update_manager/mcu/mcu_uds_peri.h"
#include "update_manager/common/function_statistics.h"
#include "update_manager/log/update_manager_logger.h"
namespace hozon {
namespace netaos {
namespace update {

UpdateManager::UpdateManager()
: stop_flag_(false)
{
}

UpdateManager::~UpdateManager()
{
}

void
UpdateManager::Init()
{
    UPDATE_LOG_I("UpdateManager::Init");
    FunctionStatistics func("UpdateManager::Init Done, ");
    UpdateSettings::Instance().Init();
    StateFileManager::Instance()->Init();
    UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);
    DownloadAgent::Instance()->Init();
    ConfigManager::Instance().Init();
    OTARecoder::Instance().Init();
    OTAStore::Instance()->Init();
    UdsCommandController::Instance()->Init();
    DiagAgent::Instance()->Init();
    OTAAgent::Instance()->Init();
    UpdateCheck::Instance().Init();
    UpdateAgent::Instance().Init();
    CmdUpgradeManager::Instance()->Init();
    SensorsUdsManager::Instance()->Init();
    OtaResultStore::Instance()->Init();
    McuUdsPeri::Instance()->Init(mcu_uds_ip, mcu_uds_port);
}

void
UpdateManager::Start()
{
    UPDATE_LOG_I("UpdateManager::Start.");
    FunctionStatistics func("UpdateManager::Start Done, ");
    ConfigManager::Instance().Start();
    UpdateAgent::Instance().Start();
    SensorsUdsManager::Instance()->Start();
    McuUdsPeri::Instance()->Start();
}

void
UpdateManager::Stop()
{
    UPDATE_LOG_I("UpdateManager::Stop");
    FunctionStatistics func("UpdateManager::Stop Done, ");
    McuUdsPeri::Instance()->Stop();
    SensorsUdsManager::Instance()->Stop();
    UpdateAgent::Instance().Stop();
    ConfigManager::Instance().Stop();
}

void
UpdateManager::Deinit()
{
    UPDATE_LOG_I("UpdateManager::Deinit");
    FunctionStatistics func("UpdateManager::Deinit Done, ");
    McuUdsPeri::Instance()->DeInit();
    OtaResultStore::Instance()->Deinit();
    SensorsUdsManager::Instance()->Deinit();
    CmdUpgradeManager::Instance()->Deinit();
    UpdateAgent::Instance().Deinit();
    UpdateCheck::Instance().Deinit();
    DiagAgent::Instance()->Deinit();
    OTAAgent::Instance()->Deinit();
    UdsCommandController::Instance()->Deinit();
    OTAStore::Instance()->Deinit();
    OTARecoder::Instance().Deinit();
    ConfigManager::Instance().Deinit();
    DownloadAgent::Instance()->Deinit();
    StateFileManager::Instance()->Deinit();
    UpdateStateMachine::Instance()->Deinit();
    UpdateSettings::Instance().Deinit();
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
