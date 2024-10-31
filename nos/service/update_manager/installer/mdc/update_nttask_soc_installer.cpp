/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: SoC installer normal task
 */

#include "update_nttask_soc_installer.h"

#include "update_cttask_interface_update.h"
#include "update_cttask_interface_activate.h"
#include "update_cttask_interface_finish.h"
#include "update_cttask_interface_wait_status.h"
#include "update_cttask_wait_sensor_update.h"
#include "update_tmtask_delay_timer.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/agent/update_agent.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/log/update_manager_logger.h"


namespace hozon {
namespace netaos {
namespace update {

#define   UPDATE_INTERFACE_RETRY_MAX_TIMES      (20*60/5)
#define   ECU_UPDATE_COMPLETED_WAIT_MAX         (6*60/5)
#define   SOC_UPDATE_RETRY_MAX_TIMES            (2)

UpdateNTTaskSocInstaller::UpdateNTTaskSocInstaller(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
            const SoC_t& soc, bool isTopTask)
    : NormalTaskBase(soc.socInfo.logicalAddr, pParent, pfnCallback, isTopTask)
    , m_soc(soc)
    , m_retryTimes(0)
    , m_updateResult(N_OK)
    , m_performance(0)
    , m_progress(0)
    , m_retry(0)
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
    }
    else {
        UPDATE_LOG_E("Task soc: %s update completed failed, performance cost %0.3lf seconds!",
            m_soc.socInfo.name.c_str(), 0.001*(GetTickCount() - m_performance));
    }
    OTARecoder::Instance().RecordFinish(m_soc.socInfo.name, result, m_progress);
}


uint32_t UpdateNTTaskSocInstaller::StartToUpdateProcess()
{
    if (!OTAAgent::Instance()->Query(m_updateStatus)) {
        return N_ERROR;
    }

    if ("kIdle" == m_updateStatus) {
        std::string curVersion;
        OTAAgent::Instance()->GetVersionInfo(curVersion);
        UPDATE_LOG_D("InterfaceUpdate query updateStatus: %s, Version current: %s, target: %s.",
            m_updateStatus.c_str(), curVersion.c_str(), m_soc.socInfo.targetVersion.c_str());
        OTARecoder::Instance().RecordUpdateVersion(m_soc.socInfo.name, curVersion, m_soc.socInfo.targetVersion);

        if (ConfigManager::Instance().IsSensorUpdate()) {
            return StartToWaitEcusUpdateCompleted();
        }

        if (curVersion == m_soc.socInfo.targetVersion && m_soc.socInfo.sameVersionCheck) {
            m_progress = 100;
            UPDATE_LOG_D("SoC current version match target version: %s, no need do update.", curVersion.c_str());
            OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "", m_progress);
            return N_OK;
        }
        return StartToInterfaceUpdate();
    }
    else if ("kReady" == m_updateStatus) {
        return StartToInterfaceActivate();
    }
    else if ("kActivated" == m_updateStatus) {
        m_progress = 100;
        return StartToInterfaceFinish();
    }
    else {
        OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Wait Status", m_progress);
        return StartToWaitActionStatus();
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToWaitEcusUpdateCompleted()
{
    // UPDATE_LOG_D("StartToWaitEcusUpdateCompleted, soc: %s!~", m_soc.socInfo.name.c_str());
    if (m_retryTimes > ECU_UPDATE_COMPLETED_WAIT_MAX) {
        UPDATE_LOG_E("wait ecus update completed, for 6 minutes limited.");
        return N_RETRY_TIMES_LIMITED;
    }

    ++m_retryTimes;
    UpdateCTTaskWaitSensorUpdate* statusTask = new UpdateCTTaskWaitSensorUpdate(this,
                                            CAST_TASK_CB(&UpdateNTTaskSocInstaller::OnWaitEcusUpdateCompletedResult));
    return post(statusTask);
}

void UpdateNTTaskSocInstaller::OnWaitEcusUpdateCompletedResult(STTask *task, uint32_t result)
{
    // UPDATE_LOG_D("OnWaitEcusUpdateCompletedResult soc: %s, result: %d!~", m_soc.socInfo.name.c_str(), result);
    if (N_OK == result) {
        m_retryTimes = 0;
        if (!OTAAgent::Instance()->Query(m_updateStatus)) {
            onCallbackResult(N_ERROR);
            return;
        }

        std::string curVersion;
        OTAAgent::Instance()->GetVersionInfo(curVersion);
        UPDATE_LOG_D("InterfaceUpdate query updateStatus: %s, Version current: %s, target: %s.",
            m_updateStatus.c_str(), curVersion.c_str(), m_soc.socInfo.targetVersion.c_str());

        if ("kIdle" == m_updateStatus) {
            if (curVersion == m_soc.socInfo.targetVersion) {
                m_progress = 100;
                UPDATE_LOG_D("SoC current version match target version: %s, no need do update.", curVersion.c_str());
                OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "", m_progress);
                onCallbackResult(N_OK);
                return;
            }
            result = StartToInterfaceUpdate();
        }
        else if ("kReady" == m_updateStatus) {
            result = StartToInterfaceActivate();
        }
        else if ("kActivated" == m_updateStatus) {
            m_progress = 100;
            result = StartToInterfaceFinish();
        }
        else {
            result = StartToWaitActionStatus();
        }
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
    UPDATE_LOG_D("OnInterfaceUpdateResult soc: %s, result: %d!~", m_soc.socInfo.name.c_str(), result);
    if (N_OK == result) {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Update", result, m_progress);
        // result = StartToInterfaceActivate();
        m_progress = 100;
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else if (N_WAIT == result) {
        OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Wait Status", m_progress);
        result = StartToWaitActionStatus();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Update", result, m_progress);
        m_updateResult = result;
        result = StartToInterfaceFinish();
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
    UPDATE_LOG_D("OnInterfaceActivateResult soc: %s, result: %d!~", m_soc.socInfo.name.c_str(), result);
    if (N_OK == result) {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Activate", result, m_progress);
        m_progress = 100;
        result = StartToInterfaceFinish();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else if (N_WAIT == result) {
        OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Wait Status", m_progress);
        result = StartToWaitActionStatus();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Activate", result, m_progress);
        m_updateResult = result;
        result = StartToInterfaceFinish();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToInterfaceFinish()
{
    UPDATE_LOG_D("StartToInterfaceFinish soc: %s!~", m_soc.socInfo.name.c_str());
    OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Finish", m_progress);
    UpdateCTTaskInterfaceFinish* finishteUpdateTask = new UpdateCTTaskInterfaceFinish(this,
                                            CAST_TASK_CB(&UpdateNTTaskSocInstaller::OnInterfaceFinishResult));
    return post(finishteUpdateTask);
}
void UpdateNTTaskSocInstaller::OnInterfaceFinishResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnInterfaceFinishResult soc: %s, result: %d, updateResult: %d.!~", m_soc.socInfo.name.c_str(), result, m_updateResult);
    OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Finish", result, m_progress);
    if (N_OK == result) {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Finish", result, m_progress);
        if (m_updateResult != N_OK) {
            result = StartToFailRetry();
        }

        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else if (N_WAIT == result) {
        OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "Interface Wait Status", m_progress);
        result = StartToWaitActionStatus();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Finish", result, m_progress);
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
    uint8_t activeProgress = 0;
    std::string updateMessage;
    std::string activeMessage;
    OTAAgent::Instance()->Query(m_updateStatus);
    OTAAgent::Instance()->GetUpdateProgress(updateProgress, updateMessage);
    OTAAgent::Instance()->GetActivationProgress(activeProgress, activeMessage);
    UPDATE_LOG_D("soc: %s, status: %s, update progress: %d, message: %s, activate progress: %d, message: %s.",
        m_soc.socInfo.name.c_str(), m_updateStatus.c_str(), updateProgress, updateMessage.c_str(), activeProgress, activeMessage.c_str());

    if(activeProgress > 0) {
        // in activate progress
        m_progress = activeProgress > m_progress ? activeProgress : m_progress;
    }
    else if (updateProgress < 100) {
        // in update progress
        m_progress = updateProgress > m_progress ? updateProgress : m_progress;
    }
    else if ("kReady" == m_updateStatus || "kBusy" == m_updateStatus || "kIdle" == m_updateStatus) {
        // update progress is 100, current still in update progress
        m_progress = updateProgress > m_progress ? updateProgress : m_progress;
    }
    else {
        // otherwise in activate progress
        m_progress = activeProgress > m_progress ? activeProgress : m_progress;
    }

    OTARecoder::Instance().RecordStepStart(m_soc.socInfo.name, "", m_progress);
    if (N_OK == result) {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Wait Status", result, m_progress);
        if ("kIdle" == m_updateStatus) {
            // update finished
        }
        else if ("kReady" == m_updateStatus) {
            if (100 == m_progress) {
                // continue to do activate
                // result = StartToInterfaceActivate();
            }
            else {
                // update action failed need do finish
                UPDATE_LOG_E("update action failed need do finish");
                m_updateResult = N_ERROR;
                result = StartToInterfaceFinish();
            }
        }
        else if ("kActivated" == m_updateStatus) {
            UPDATE_LOG_D("update activte completed continue do finish");
            result = StartToInterfaceFinish();
        }
        else {
            result = N_ERROR;
        }

        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else if (N_WAIT == result) {
        result = StartToWaitActionStatus();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        OTARecoder::Instance().RecordStepFinish(m_soc.socInfo.name, "Interface Wait Status", result, m_progress);
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSocInstaller::StartToFailRetry()
{
    ++m_retry;
    if (m_retry >= SOC_UPDATE_RETRY_MAX_TIMES) {
        UPDATE_LOG_E("soc: %s update retry failed, for %d times limited.", m_soc.socInfo.name.c_str(), SOC_UPDATE_RETRY_MAX_TIMES);
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
    UPDATE_LOG_D("OnFailRetryResult retry: %d, delay: %d ms, soc: %s, result: %d, progress: %d%% !~",
        m_retry, OTA_TIMER_P3_CLIENT_PYH, m_soc.socInfo.name.c_str(), result, m_progress);
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


}  // namespace update
}  // namespace netaos
}  // namespace hozon
