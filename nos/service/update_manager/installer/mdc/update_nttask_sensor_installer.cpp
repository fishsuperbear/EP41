/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: sensor update install task
 */

#include "update_nttask_sensor_installer.h"

#include "update_nttask_transfer_file.h"
#include "update_nttask_security_access.h"
#include "update_nttask_send_commands.h"
#include "update_tmtask_delay_timer.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/record/ota_record.h"

namespace hozon {
namespace netaos {
namespace update {

#define   SENSOR_UPDATE_RETRY_MAX_TIMES         (3)

UpdateNTTaskSensorInstaller::UpdateNTTaskSensorInstaller(NormalTaskBase* pParent, STObject::TaskCB pfnCallback, const Sensor_t& sensor, bool isTopTask)
    : NormalTaskBase(sensor.sensorInfo.logicalAddr, pParent, pfnCallback, isTopTask)
    , index_(0)
    , sensor_(sensor)
    , performance_(0)
    , progress_(0)
    , retry_(0)
{
}

UpdateNTTaskSensorInstaller::~UpdateNTTaskSensorInstaller()
{
}

uint32_t
UpdateNTTaskSensorInstaller::doAction()
{
    OTARecoder::Instance().RecordStart(sensor_.sensorInfo.name, progress_);
    performance_ = GetTickCount();
    return StartToUpdateProcess();
}

void
UpdateNTTaskSensorInstaller::onCallbackAction(uint32_t result)
{
    if (eOK == result) {
        UPDATE_LOG_D("Task sensor: %s update completed successful, performance cost %0.3lf seconds!",
            sensor_.sensorInfo.name.c_str(), 0.001*(GetTickCount() - performance_));
        progress_ = 100;
    }
    else {
        UPDATE_LOG_E("Task sensor: %s update completed failed, performance cost %0.3lf seconds!",
            sensor_.sensorInfo.name.c_str(), 0.001*(GetTickCount() - performance_));
    }
    OTARecoder::Instance().RecordFinish(sensor_.sensorInfo.name, result, progress_);
}

uint32_t
UpdateNTTaskSensorInstaller::StartToUpdateProcess()
{
    UPDATE_LOG_D("StartToUpdateProcess!~ sensor: %s, logicalAddr: %X, updateType: %d, process: %d / total: %ld, transType: %d",
        sensor_.sensorInfo.name.c_str(), sensor_.sensorInfo.logicalAddr, sensor_.sensorInfo.updateType, index_ + 1, sensor_.process.size(), sensor_.process[index_].transType);
    // typedef struct {
    //     uint8_t transType;  // 0: none, 1: buff data, 2: security, 3: transfer file
    //     uint8_t idType;     // 0: none, 1: physical 2: functional
    //     uint32_t waitTime;  // trans data before need to delay
    //     std::vector<uint8_t> transData;  // transfer data
    //     uint32_t transDataSize;  // 0: no check, other: may need add transData
    //     std::vector<uint8_t> recvExpect; // expect recv data
    //     uint8_t algoType;  // 0: none, 1: applevel, 2: boot level, 3：md5 checksum for file
    //     uint32_t mask;     // 4 bytes app mask or boot mask
    //     std::string filePath; // "": invalid or not used
    // } UpdateCase_t;

    uint32_t ret = N_ERROR;
    if (sensor_.process[index_].transType == 1) {
        // uint8_t transType;  // 0: none, 1: buff data, 2: security, 3: transfer file
        if (sensor_.process[index_].transData.size() == 0) {
            UPDATE_LOG_E("process script invalid, transfer data size is 0");
            return ret;
        }
        ret = StartToSendCommand();
    }
    else if (sensor_.process[index_].transType == 2) {
        // uint8_t transType;  // 0: none, 1: buff data, 2: security, 3: transfer file
        ret = StartToSecurityAccess();
    }
    else if (sensor_.process[index_].transType == 3) {
        // uint8_t transType;  // 0: none, 1: buff data, 2: security, 3: transfer file
        if (access(sensor_.process[index_].filePath.c_str(), F_OK)) {
            UPDATE_LOG_E("update file is not existed, expect file: %s.", sensor_.process[index_].filePath.c_str());
            return ret;
        }
        ret = StartToTransFile();
    }
    return ret;
}

uint32_t UpdateNTTaskSensorInstaller::StartToSendCommand()
{
    if ((sensor_.process)[index_].beginProgress == 0 || (sensor_.process)[index_].beginProgress < progress_) {
        (sensor_.process)[index_].beginProgress = progress_;
    }
    else {
        progress_ = (sensor_.process)[index_].beginProgress;
    }
    if ((sensor_.process)[index_].endProgress == 0 || (sensor_.process)[index_].endProgress < progress_) {
        (sensor_.process)[index_].endProgress = progress_ + 1 >= 100 ? 100 : progress_ + 1;
    }
    UPDATE_LOG_D("StartToSendCommand sensor: %s, process: %d / total process: %ld, progress: %d%% !~",
        sensor_.sensorInfo.name.c_str(), index_ + 1, sensor_.process.size(), progress_);
    OTARecoder::Instance().RecordStepStart(sensor_.sensorInfo.name, (sensor_.process)[index_].updateStep, progress_);
    UpdateNTTaskSendCommands* commandsTask = new UpdateNTTaskSendCommands(this,
                                            CAST_TASK_CB(&UpdateNTTaskSensorInstaller::OnSendCommandResult),
                                            sensor_.sensorInfo,
                                            (sensor_.process)[index_],
                                            false);
    return post(commandsTask);
}

void UpdateNTTaskSensorInstaller::OnSendCommandResult(STTask* task, uint32_t result)
{
    progress_ = (sensor_.process)[index_].endProgress;
    UPDATE_LOG_D("OnSendCommandResult sensor: %s, result: %d, process: %d / total: %ld, progress: %d%% !~",
        sensor_.sensorInfo.name.c_str(), result, index_ + 1, sensor_.process.size(), progress_);
    OTARecoder::Instance().RecordStepFinish(sensor_.sensorInfo.name, (sensor_.process)[index_].updateStep, result, progress_);
    ++index_;
    if (N_OK == result) {
        if (index_ >= sensor_.process.size()) {
            // all task completed.
            UPDATE_LOG_D("all task completed, sensor: %s!~", sensor_.sensorInfo.name.c_str());
            onCallbackResult(result);
            return;
        }

        UpdateNTTaskSendCommands* sftask = static_cast<UpdateNTTaskSendCommands*>(task);
        if (nullptr == sftask) {
            result = N_ERROR;
            onCallbackResult(result);
            return;
        }

        if (sftask->GetResInfo().resContent.size() > 3 && sftask->GetResInfo().resContent[0] == 0x62) {
            std::vector<uint8_t> resContent = sftask->GetResInfo().resContent;
            if (resContent[1] == 0xF1 && resContent[2] == 0xC0) {
                // Read Software Version $62 F1 C0
                bool match = true;
                if (resContent.size() - 3 < sensor_.sensorInfo.targetVersion.size()) {
                    match = false;
                }
                for (uint8_t index = 0; index < sensor_.sensorInfo.targetVersion.size(); ++index) {
                    if (sensor_.sensorInfo.targetVersion[index] != resContent[index + 3]) {
                        match = false;
                        break;
                    }
                }
                std::string recvVersion = std::string(reinterpret_cast<const char*>(&resContent[3]), resContent.size() - 3);
                UPDATE_LOG_D("SoftwareVersion sameCheck: %d, target: %s, recv: %s", match, sensor_.sensorInfo.targetVersion.c_str(), recvVersion.c_str());
                OTARecoder::Instance().RecordUpdateVersion(sensor_.sensorInfo.name, recvVersion, sensor_.sensorInfo.targetVersion);

                if (match && sensor_.sensorInfo.sameVersionCheck) {
                    result = N_OK;
                    progress_ = 100;
                    onCallbackResult(result);
                    return;
                }
            }
        }

        result = StartToUpdateProcess();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        result = StartToFailRetry();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSensorInstaller::StartToSecurityAccess()
{
    if ((sensor_.process)[index_].beginProgress == 0 || (sensor_.process)[index_].beginProgress < progress_) {
        (sensor_.process)[index_].beginProgress = progress_;
    }
    else {
        progress_ = (sensor_.process)[index_].beginProgress;
    }
    if ((sensor_.process)[index_].endProgress == 0 || (sensor_.process)[index_].endProgress < progress_) {
        (sensor_.process)[index_].endProgress = progress_ + 2 >= 100 ? 100 : progress_ + 2;
    }
    UPDATE_LOG_D("StartToSecurityAccess sensor: %s, process: %d / total: %ld, progress: %d%% !~",
        sensor_.sensorInfo.name.c_str(), index_ + 1, sensor_.process.size(), progress_);
    OTARecoder::Instance().RecordStepStart(sensor_.sensorInfo.name, (sensor_.process)[index_].updateStep, progress_);
    UpdateNTTaskSecurityAccess* secureTask = new UpdateNTTaskSecurityAccess(this,
                                            CAST_TASK_CB(&UpdateNTTaskSensorInstaller::OnSecurityAccessResult),
                                            sensor_.sensorInfo,
                                            (sensor_.process)[index_],
                                            false);
    return post(secureTask);
}

void UpdateNTTaskSensorInstaller::OnSecurityAccessResult(STTask *task, uint32_t result)
{
    progress_ = (sensor_.process)[index_].endProgress;
    UPDATE_LOG_D("OnSecurityAccessResult sensor: %s, result: %d, process: %d / total: %ld, progress: %d%% !~",
        sensor_.sensorInfo.name.c_str(), result, index_ + 1, sensor_.process.size(), progress_);
    OTARecoder::Instance().RecordStepFinish(sensor_.sensorInfo.name, (sensor_.process)[index_].updateStep, result, progress_);
    ++index_;
    if (N_OK == result) {
        if (index_ >= sensor_.process.size()) {
            // all task completed.
            onCallbackResult(result);
            return;
        }

        result = StartToUpdateProcess();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        result = StartToFailRetry();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSensorInstaller::StartToTransFile()
{
    if ((sensor_.process)[index_].beginProgress == 0 || (sensor_.process)[index_].beginProgress < progress_) {
        (sensor_.process)[index_].beginProgress = progress_;
    }
    else {
        progress_ = (sensor_.process)[index_].beginProgress;
    }
    if ((sensor_.process)[index_].endProgress == 0 || (sensor_.process)[index_].endProgress < progress_) {
        (sensor_.process)[index_].endProgress = progress_ + 1 >= 100 ? 100 : progress_ + 1;
    }
    UPDATE_LOG_D("StartToTransFile sensor: %s, process: %d / total: %ld, progress: %d%% !~",
        sensor_.sensorInfo.name.c_str(), index_ + 1, sensor_.process.size(), progress_);
    OTARecoder::Instance().RecordStepStart(sensor_.sensorInfo.name, (sensor_.process)[index_].updateStep, progress_);
    UpdateNTTaskTransferFile* fileTask = new UpdateNTTaskTransferFile(this,
                                        CAST_TASK_CB(&UpdateNTTaskSensorInstaller::OnTransFileResult),
                                        sensor_.sensorInfo,
                                        (sensor_.process)[index_],
                                        false);
    return post(fileTask);
}

void UpdateNTTaskSensorInstaller::OnTransFileResult(STTask *task, uint32_t result)
{
    UpdateNTTaskTransferFile* sftask = static_cast<UpdateNTTaskTransferFile*>(task);
    if (nullptr != sftask) {
        progress_ = sftask->GetProgress();
    }
    UPDATE_LOG_D("OnTransFileResult sensor: %s, result: %d, process: %d / total: %ld, progress: %d%% !~",
        sensor_.sensorInfo.name.c_str(), result, index_ + 1, sensor_.process.size(), progress_);
    OTARecoder::Instance().RecordStepFinish(sensor_.sensorInfo.name, (sensor_.process)[index_].updateStep, result, progress_);
    ++index_;
    if (N_OK == result) {
        if (index_ >= sensor_.process.size()) {
            // all task completed.
            onCallbackResult(result);
            return;
        }

        result = StartToUpdateProcess();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        result = StartToFailRetry();
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSensorInstaller::StartToFailRetry()
{
    ++retry_;
    if (retry_ >= SENSOR_UPDATE_RETRY_MAX_TIMES) {
        UPDATE_LOG_E("sensor: %s update retry failed, for %d times limited.", sensor_.sensorInfo.name.c_str(), SENSOR_UPDATE_RETRY_MAX_TIMES);
        return N_RETRY_TIMES_LIMITED;
    }

    UPDATE_LOG_D("StartToFailRetry retry: %d, delay: %d ms, sensor: %s, process: %d / total: %ld, progress: %d%% !~",
        retry_, OTA_TIMER_P3_CLIENT_PYH, sensor_.sensorInfo.name.c_str(), index_ + 1, sensor_.process.size(), progress_);
    UpdateTMTaskDelayTimer* task = new UpdateTMTaskDelayTimer(TIMER_UPDATE_RETRY_DELAY,
                                    this,
                                    CAST_TASK_CB(&UpdateNTTaskSensorInstaller::OnFailRetryResult),
                                    OTA_TIMER_P3_CLIENT_PYH);
    return post(task);
}

void UpdateNTTaskSensorInstaller::OnFailRetryResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnFailRetryResult retry: %d, delay: %d ms, sensor: %s, result: %d, process: %d / total: %ld, progress: %d%% !~",
        retry_, OTA_TIMER_P3_CLIENT_PYH, sensor_.sensorInfo.name.c_str(), result, index_ + 1, sensor_.process.size(), progress_);
    if (N_OK == result) {
        index_ = 4;  // sensor failed retry no need check version and ecu info
        progress_ = 0;
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

uint32_t UpdateNTTaskSensorInstaller::GetTickCount()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
