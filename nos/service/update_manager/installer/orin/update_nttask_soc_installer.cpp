/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: SoC installer normal task
 */

#include "update_nttask_soc_installer.h"
#include "update_cttask_interface_update.h"
#include "update_cttask_interface_activate.h"
#include "update_cttask_interface_wait_status.h"
#include "update_cttask_wait_sensor_update.h"
#include "update_tmtask_delay_timer.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/agent/update_agent.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/state_machine/state_file_manager.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "update_manager/manager/uds_command_controller.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/update_check/update_check.h"
#include "update_manager/record/ota_result_store.h"
#include "update_manager/common/um_functions.h"
#include "update_manager/log/update_manager_logger.h"
namespace hozon {
namespace netaos {
namespace update {

#define   UPDATE_INTERFACE_RETRY_MAX_TIMES      (30*60/5)
#define   ECU_UPDATE_COMPLETED_WAIT_MAX         (10*60/5)

UpdateNTTaskSocInstaller::UpdateNTTaskSocInstaller(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
            const SoC_t& soc, bool isTopTask)
    : NormalTaskBase(soc.socInfo.logicalAddr, pParent, pfnCallback, isTopTask)
    , m_soc(soc)
    , m_retryTimes(0)
    , m_updateResult(N_OK)
    , m_performance(0)
    , m_progress(0)
    , m_retry(0)
    , m_update_flag(true)
{
}

UpdateNTTaskSocInstaller::~UpdateNTTaskSocInstaller()
{
}

uint32_t UpdateNTTaskSocInstaller::doAction()
{
    OTARecoder::Instance().RecordStart(m_soc.socInfo.name, m_progress);
    m_performance = GetTickCount();
    return StartToUpdateProcess();
}

void UpdateNTTaskSocInstaller::onCallbackAction(uint32_t result)
{
    if (eOK == result) {
        UPDATE_LOG_D("Task soc: %s update completed successful, performance cost %0.3lf seconds!",
            m_soc.socInfo.name.c_str(), 0.001*(GetTickCount() - m_performance));
        m_progress = 100;
        if (m_update_flag) {
            OtaResultStore::Instance()->UpdateResult("ORIN", "Succ");
            UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATED);
            // 启动timer 30min超时
            UpdateCheck::Instance().StartTimer(30*60);
            // 识别cmd升级，触发切面，重启动作
            if (CmdUpgradeManager::Instance()->IsCmdTriggerUpgrade()) {
                UPDATE_LOG_D("Update complete! now to switch slot && reboot!");
                // Soc版本一致的case，无须触发切面
                if (!UdsCommandController::Instance()->IsSocVersionSame()) {
                    UPDATE_LOG_D("Call OTAAgent SwitchSlot");
                    OTAAgent::Instance()->SwitchSlot();
                }
                std::string targetVer{};
                ConfigManager::Instance().GetMajorVersion(targetVer);
                std::vector<uint8_t> version{targetVer.begin(), targetVer.end()};
                OTAStore::Instance()->WriteECUVersionData(version);
                OTAStore::Instance()->WriteECUSWData(version);
                OTARecoder::Instance().RecordFinish(m_soc.socInfo.name, result, m_progress);
                SystemSync();
                std::this_thread::sleep_for(std::chrono::seconds(2));
                SystemSync();
                if (UdsCommandController::Instance()->IsSocVersionSame()) {
                    UM_DEBUG << "update complete, now wait to reboot!";
                    OTAAgent::Instance()->HardReboot();
                } else {
                    UdsCommandController::Instance()->SetVersionSameFlag(false);
                    UM_DEBUG << "sync && reboot";
                    OTAAgent::Instance()->Reboot();
                }
            } else {
                OTARecoder::Instance().RecordFinish(m_soc.socInfo.name, result, m_progress);
            }
        } else {
            UpdateStateMachine::Instance()->SwitchState(State::OTA_ACTIVED);
            m_update_flag = true;
            OTARecoder::Instance().RecordFinish(m_soc.socInfo.name, result, m_progress);
        }
    }
    else {
        UPDATE_LOG_E("Task soc: %s update completed failed, performance cost %0.3lf seconds!",
            m_soc.socInfo.name.c_str(), 0.001*(GetTickCount() - m_performance));
        OtaResultStore::Instance()->UpdateResult("ORIN", "Fail");
        uint8_t updateProgress = 0;
        std::string updateMessage{};
        OTAAgent::Instance()->GetUpdateProgress(updateProgress, updateMessage);
        UPDATE_LOG_D("soc: %s, status: %s, update progress: %d .", m_soc.socInfo.name.c_str(), m_updateStatus.c_str(), updateProgress);
        if (static_cast<uint16_t>(updateProgress) <= 49) {
            UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::SOC_UPDATE_FAILED);
        } else {
            UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::MCU_UPDATE_FAILED);
        }
        DiagAgent::Instance()->ResetTotalProgress();
        ConfigManager::Instance().ClearSensorUpdate();
        OTARecoder::Instance().RecordFinish(m_soc.socInfo.name, result, m_progress);
    }
}


uint32_t UpdateNTTaskSocInstaller::StartToUpdateProcess()
{
    // 德赛不会返回，需要从自己数据库读, 从而判断是否进入激活
    std::string curState{};
    StateFileManager::Instance()->ReadStateFile(curState);
    UPDATE_LOG_D("UpdateNTTaskSocInstaller query current storage state is : %s", curState.c_str());
    m_update_flag = true;
    if (curState == "OTA_ACTIVING") {
        // 1101 后的动作，做激活（实则是版本校验，cfg读取的信息和当前版本信息）
        m_progress = 100;
        m_update_flag = false;
        return StartToInterfaceActivate();
    }
    if (!OTAAgent::Instance()->Query(m_updateStatus)) {
        return N_ERROR;
    }
    UPDATE_LOG_D("InterfaceUpdate query updateStatus: %s", m_updateStatus.c_str());
    if ("IDLE" == m_updateStatus) {

        if (ConfigManager::Instance().IsSensorUpdate()) {
            return StartToWaitEcusUpdateCompleted();
        }

        auto versionRes = VersionCheck();
        if (versionRes) {
            m_progress = 100;
            UdsCommandController::Instance()->SetVersionSameFlag(true);
            UPDATE_LOG_W("SoC current Dsv & Soc & Mcu version is same, no need to update");
            OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "", m_progress);
            return N_OK;
        }
        return StartToInterfaceUpdate();
    }
    else {
        // 此时返回UPDATE_FAILED / UPDATING，认为升级失败
        m_updateResult = N_ERROR;
        return N_ERROR;
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToWaitEcusUpdateCompleted()
{
    UPDATE_LOG_D("StartToWaitEcusUpdateCompleted, soc: %s!~", m_soc.socInfo.name.c_str());
    if (m_retryTimes > ECU_UPDATE_COMPLETED_WAIT_MAX) {
        UPDATE_LOG_E("wait ecus update completed, for 10 minutes limited.");
        return N_RETRY_TIMES_LIMITED;
    }

    ++m_retryTimes;
    UpdateCTTaskWaitSensorUpdate* statusTask = new UpdateCTTaskWaitSensorUpdate(this,
                                            CAST_TASK_CB(&UpdateNTTaskSocInstaller::OnWaitEcusUpdateCompletedResult));
    return post(statusTask);
}

void UpdateNTTaskSocInstaller::OnWaitEcusUpdateCompletedResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnWaitEcusUpdateCompletedResult soc: %s, result: %s!~", m_soc.socInfo.name.c_str(), GetTaskResultString(result).c_str());
    if (N_OK == result) {
        m_retryTimes = 0;
        auto versionRes = VersionCheck();
        if (versionRes) {
            m_progress = 100;
            UdsCommandController::Instance()->SetVersionSameFlag(true);
            UPDATE_LOG_D("SoC current Dsv & Soc & Mcu version is same, no need to update");
            OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "", m_progress);
            onCallbackResult(N_OK);
            return;
        }
        result = StartToInterfaceUpdate();

        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else if (N_WAIT == result) {
        result = StartToWaitEcusUpdateCompleted();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToInterfaceUpdate()
{
    UPDATE_LOG_D("StartToInterfaceUpdate soc: %s, package: %s!~", m_soc.socInfo.name.c_str(), m_soc.socInfo.firmwareName.c_str());
    OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Update", m_progress);
    UpdateCTTaskInterfaceUpdata* interfaceUpdateTask = new UpdateCTTaskInterfaceUpdata(this,
                                            CAST_TASK_CB(&UpdateNTTaskSocInstaller::OnInterfaceUpdateResult),
                                            m_soc.socInfo.firmwareName);
    return post(interfaceUpdateTask);
}

void UpdateNTTaskSocInstaller::OnInterfaceUpdateResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnInterfaceUpdateResult soc: %s, result: %s!~", m_soc.socInfo.name.c_str(), GetTaskResultString(result).c_str());
    if (N_OK == result) {
        // 升级结束
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Update", result, m_progress);
        m_progress = 100;
    
        std::string targetSocVer{m_soc.socInfo.targetVersion};
        OTAStore::Instance()->WriteSocVersionData(targetSocVer);
        std::string targetMcuVersion{};
        ConfigManager::Instance().GetMcuVersion(targetMcuVersion);
        OTAStore::Instance()->WriteMcuVersionData(targetMcuVersion);
        std::string targetDsvVersion{};
        ConfigManager::Instance().GetDsvVersion(targetDsvVersion);
        OTAStore::Instance()->WriteDsvVersionData(targetDsvVersion);

        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else if (N_WAIT == result) {
        // 升级中
        OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Wait Status", m_progress);
        result = StartToWaitActionStatus();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        // 升级失败
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Update", result, m_progress);
        m_updateResult = result;
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToInterfaceActivate()
{
    UPDATE_LOG_D("StartToInterfaceActivate soc: %s!~", m_soc.socInfo.name.c_str());
    OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Activate", m_progress);
    UpdateCTTaskInterfaceActivate* activateUpdateTask = new UpdateCTTaskInterfaceActivate(this,
                                            CAST_TASK_CB(&UpdateNTTaskSocInstaller::OnInterfaceActivateResult));
    return post(activateUpdateTask);
}
void UpdateNTTaskSocInstaller::OnInterfaceActivateResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnInterfaceActivateResult soc: %s, result: %s!~", m_soc.socInfo.name.c_str(), GetTaskResultString(result).c_str());
    if (N_OK == result) {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Activate", result, m_progress);
        m_progress = 100;
        if (eContinue != result) {
            onCallbackResult(result);
        }
    } else {
        UPDATE_LOG_E("active error, result is : %d.", result);
        // 激活进度更新为0
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Activate", result, m_progress);
        m_updateResult = result;
        m_progress = 0;
        result = StartToFailRetry();


        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToWaitActionStatus()
{
    if (m_retryTimes > UPDATE_INTERFACE_RETRY_MAX_TIMES) {
        UPDATE_LOG_E("wait update some action failed, for 20 minutes limited.");
        return N_RETRY_TIMES_LIMITED;
    }

    ++m_retryTimes;
    UpdateCTTaskInterfaceWaitStatus* statusTask = new UpdateCTTaskInterfaceWaitStatus(this,
                                            CAST_TASK_CB(&UpdateNTTaskSocInstaller::OnWaitActionStatusResult));
    return post(statusTask);
}

void UpdateNTTaskSocInstaller::OnWaitActionStatusResult(STTask *task, uint32_t result)
{
    uint8_t updateProgress = 0;
    std::string updateMessage;
    OTAAgent::Instance()->Query(m_updateStatus);
    OTAAgent::Instance()->GetUpdateProgress(updateProgress, updateMessage);
    UPDATE_LOG_D("soc: %s, status: %s, update progress: %d .", m_soc.socInfo.name.c_str(), m_updateStatus.c_str(), updateProgress);

    if (updateProgress < 100) {
        // in update progress
        m_progress = updateProgress > m_progress ? updateProgress : m_progress;
    }
    else if ("UPDATE_SUCCESS" == m_updateStatus) {
        // update progress is 100, current still in update progress
        m_progress = updateProgress > m_progress ? updateProgress : m_progress;
    }

    OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "", m_progress);
    if (N_OK == result) {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Wait Status", result, m_progress);
        if ("UPDATE_SUCCESS" == m_updateStatus) {
            m_progress = 100;
            if (100 == m_progress) {
                // 升级完成
                m_updateResult = N_OK;
                
                std::string targetSocVer{m_soc.socInfo.targetVersion};
                OTAStore::Instance()->WriteSocVersionData(targetSocVer);
                std::string targetMcuVersion{};
                ConfigManager::Instance().GetMcuVersion(targetMcuVersion);
                OTAStore::Instance()->WriteMcuVersionData(targetMcuVersion);
                std::string targetDsvVersion{};
                ConfigManager::Instance().GetDsvVersion(targetDsvVersion);
                OTAStore::Instance()->WriteDsvVersionData(targetDsvVersion);
            }
            else {
                // update action failed need do finish
                UPDATE_LOG_E("update action failed need do finish");
                m_updateResult = N_ERROR;
            }
        }
        else {
            result = N_ERROR;
        }
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else if (N_WAIT == result) {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Wait Status", result, m_progress);
        result = StartToWaitActionStatus();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Wait Status", result, m_progress);
        UM_DEBUG << "Soc Update Failed ,now to restart UpdateService and retry!";
        m_updateResult = result;
        auto res = UpdateCheck::Instance().RestartUpdateService();
        if (res) {
            UM_DEBUG << "start to retry!";
            result = StartToFailRetry(2);
        }
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToFailRetry(uint16_t retryTimes)
{
    ++m_retry;
    if (m_retry >= retryTimes) {
        UPDATE_LOG_E("soc: %s update retry failed, for %d times limited.", m_soc.socInfo.name.c_str(), retryTimes);
        return N_RETRY_TIMES_LIMITED;
    }

    UPDATE_LOG_D("StartToFailRetry retry: %d, delay: %d ms, soc: %s, progress: %d%% !~",
        m_retry, OTA_TIMER_P3_CLIENT_PYH, m_soc.socInfo.name.c_str(), m_progress);
    UpdateTMTaskDelayTimer* task = new UpdateTMTaskDelayTimer(TIMER_UPDATE_RETRY_DELAY,
                                    this,
                                    CAST_TASK_CB(&UpdateNTTaskSocInstaller::OnFailRetryResult),
                                    OTA_TIMER_P3_CLIENT_PYH);
    return post(task);
}

void UpdateNTTaskSocInstaller::OnFailRetryResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnFailRetryResult retry: %d, delay: %d ms, soc: %s, result: %s, progress: %d%% !~",
        m_retry, OTA_TIMER_P3_CLIENT_PYH, m_soc.socInfo.name.c_str(), GetTaskResultString(result).c_str(), m_progress);
    if (N_OK == result) {
        m_progress = 0;
        m_updateResult = N_OK;
        result = StartToUpdateProcess();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSocInstaller::GetTickCount()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

bool UpdateNTTaskSocInstaller::VersionCheck()
{
    if (m_soc.socInfo.sameVersionCheck) {
        // go on
    } else {
        return false;
    }
    std::string curSocVersion{};
    OTAStore::Instance()->ReadSocVersionData(curSocVersion);
    UPDATE_LOG_D("curSocVersion from desay is %s, targetSocVersion from pkg is %s.", curSocVersion.c_str(), m_soc.socInfo.targetVersion.c_str());
    OTARecoder::Instance().RecordUpdateVersion(m_soc.socInfo.name, curSocVersion, m_soc.socInfo.targetVersion);

    std::string curDsvVersion{};
    OTAAgent::Instance()->GetVersionInfo(curDsvVersion);
    curDsvVersion.erase(curDsvVersion.find_last_not_of('\0') + 1);
    std::string targetDsvVersion{};
    ConfigManager::Instance().GetDsvVersion(targetDsvVersion);
    UPDATE_LOG_D("curDsvVersion from desay is %s, targetDsvVersion from pkg is %s.", curDsvVersion.c_str(), targetDsvVersion.c_str());

    std::string curMcuVersion{};
    OTAStore::Instance()->ReadMcuVersionData(curMcuVersion);
    std::string targetMcuVersion{};
    ConfigManager::Instance().GetMcuVersion(targetMcuVersion);
    UPDATE_LOG_D("curMcuVersion from cfg is %s, targetMcuVersion from pkg is %s.", curMcuVersion.c_str(), targetMcuVersion.c_str());

    if (curSocVersion == m_soc.socInfo.targetVersion && curDsvVersion == targetDsvVersion && curMcuVersion == targetMcuVersion) {
        if (UpdateSettings::Instance().SameVersionUpdate()) {
            UM_DEBUG << "same version force update!";
            return false;
        } else {
            return true;
        }
    }
    return false;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
