/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: UA module
 */
#include "update_manager/agent/update_agent.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_nttask_sensor_installer.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/config/config_manager.h"
#include "update_nttask_soc_installer.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/state_machine/state_file_manager.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "update_manager/state_machine/update_state_machine.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace std;
using namespace placeholders;


UpdateAgent::UpdateAgent() : update_success_flag_(0)
{
}

int32_t
UpdateAgent::Init(void)
{
    UM_INFO << "UpdateAgent::Init.";
    DiagAgent::Instance()->RegistUdsRawDataReceiveCallback([this](const std::unique_ptr<uds_raw_data_resp_t>& uds_data) -> void {
        UPDATE_LOG_D("Uds receive callback updateType: %d, reqSa: %X, reqTa: %X, result: %d, uds data size: %ld.", uds_data->bus_type,
            uds_data->sa, uds_data->ta, uds_data->result, uds_data->data_vec.size());
            uint32_t eventType = uds_data->bus_type;
            uint32_t eventId = uds_data->sa;
            int32_t evtVal1 = uds_data->ta;
            int32_t evtVal2 = uds_data->result;
            UpdateTaskEvent *event = new UpdateTaskEvent(eventType, eventId, evtVal1, evtVal2, uds_data->data_vec);
            post(event);
        }
    );
    init();
    UM_INFO << "UpdateAgent::Init Done.";
    return 0;
}

int32_t UpdateAgent::Start(void)
{
    UM_INFO << "UpdateAgent::Start.";
    start();


#ifdef BUILD_FOR_MDC
    std::string updateStatus = "unknown";
    OTAAgent::Instance()->Query(updateStatus);
    UPDATE_LOG_D("OTAAgent query soc status: %s.", updateStatus.c_str());
    if ("kVerifying" == updateStatus || "kActivating" == updateStatus || "kActivated" == updateStatus ) 
#else
    // 读最新state，判断是否进入激活阶段
    std::string curState{};
    StateFileManager::Instance()->ReadStateFile(curState);
    UPDATE_LOG_D("OTAAgent query current storage state is : %s", curState.c_str());
    if ("OTA_UPDATED" == curState) 
#endif
    {
        UpdateStateMachine::Instance()->SetInitialState(State::OTA_ACTIVING);
        // update is after reboot, need continue update finish action
        UPDATE_LOG_D("after update power off restart continue do update task.");
        SocManifest_t socManifests = ConfigManager::Instance().GetSocManifest();
        // 配置文件被意外删除，此处做处理，防止卡在activing状态
        if (socManifests.socs.empty())
        {
            UPDATE_LOG_D("parse xml failed ,no soc to active !");
            UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::CONFIG_DELETED);
        }
        
        for (auto iter : socManifests.socs) {
            OTARecoder::Instance().AddSocProgress(iter.socInfo.name, iter.socInfo.targetVersion);
            UpdateNTTaskSocInstaller* task = new UpdateNTTaskSocInstaller(nullptr, nullptr, iter, true);
            int32_t ret = UpdateAgent::Instance().post(task);
            if (eContinue != ret) {
                UPDATE_LOG_E("post update continue task fail, ret = %d", ret);
            }
            else {
                UPDATE_LOG_D("post update continue task(%p) success, ret = %d(eContinue)", task, ret);
            }
        }
    }
    UM_INFO << "UpdateAgent::Start Done.";
    return 0;
}

int32_t UpdateAgent::Stop(void)
{
    UM_INFO << "UpdateAgent::Stop.";
    stop();
    UM_INFO << "UpdateAgent::Stop Done.";
    return 0;
}

int32_t UpdateAgent::Deinit(void)
{
    UM_INFO << "UpdateAgent::Deinit.";
    deinit();
    UM_INFO << "UpdateAgent::Deinit Done.";
    return 0;
}

uint8_t
UpdateAgent::Update()
{
    UPDATE_LOG_D("Update all in the update package.");
    SensorManifest_t sensorManifests = ConfigManager::Instance().GetSensorManifest();
    SocManifest_t socManifests = ConfigManager::Instance().GetSocManifest();

    auto res = CmdUpgradeManager::Instance()->GetEcuMode();
    // 声明一个multimap 存入{sequence  sensorInfo}键值对
    std::multimap<uint8_t, Sensor_t> sensorSequenceMap{};
    for (auto it : sensorManifests.sensors) {
        if (res == "DEFAULT" || res == it.sensorInfo.name) {
            sensorSequenceMap.insert(std::make_pair(it.sensorInfo.updateSequence, it));
        }
    }
    // 当只升级sensor，但sensor 不存在，需要切换到UPDATE_FAILED
    if (res != "DEFAULT" && sensorSequenceMap.empty()) {
        UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::SENSOR_NOT_EXIST);
        return 0;
    }
    for (auto it = sensorSequenceMap.begin(); it != sensorSequenceMap.end(); it++)
    {
        if (it->second.process.size() > 0) {
            // docan or doip to update, 1: docan , 2: doip
            OTARecoder::Instance().AddSensorProgress(it->second.sensorInfo.name, "00.00.00");
        }
    }
    for (auto iter : socManifests.socs) {
        OTARecoder::Instance().AddSocProgress(iter.socInfo.name, iter.socInfo.targetVersion);
    }
    std::vector<std::string> SensorVec{};
    for (auto it = sensorSequenceMap.begin(); it != sensorSequenceMap.end(); it++)
    {
        if (it->second.process.size() > 0) {
            UpdateNTTaskSensorInstaller* task = new UpdateNTTaskSensorInstaller(nullptr, nullptr, it->second, true);
            int32_t ret = UpdateAgent::Instance().post(task);
            SensorVec.emplace_back(it->second.sensorInfo.name);
            if (eContinue != ret) {
                UPDATE_LOG_E("post sensor: %s, updateType: %d, update task fail, ret = %d",
                    it->second.sensorInfo.name.c_str(), it->second.sensorInfo.updateType, ret);
            }
            else {
                UPDATE_LOG_D("post sensor: %s, updateType: %d, task(%p) success, ret = %d(eContinue)",
                    it->second.sensorInfo.name.c_str(), it->second.sensorInfo.updateType, task, ret);
            }
            // 如果seq相同，则一起 postTask ，否则等待seq靠前的sensor升级结束
            if (std::next(it) != sensorSequenceMap.end() && it->first == std::next(it)->first)
            {
                continue;
            }
            // 当前为最后一个元素，则退出循环
            if (std::next(it) == sensorSequenceMap.end())
            {
                break;
            }
            // 这里一定会退出，内部task有超时时间，换句话说IsSensorUpdateCompleted()一定时间内必会返回true
            while(!OTARecoder::Instance().IsSensorUpdateCompleted(SensorVec))
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }

    for (auto iter : socManifests.socs) {
        OTARecoder::Instance().AddSocProgress(iter.socInfo.name, iter.socInfo.targetVersion);
        UpdateNTTaskSocInstaller* task = new UpdateNTTaskSocInstaller(nullptr, nullptr, iter, true);
        int32_t ret = UpdateAgent::Instance().post(task);
        if (eContinue != ret) {
            UPDATE_LOG_E("post soc: %s, updateType: %d, update continue task fail, ret = %d",
                iter.socInfo.name.c_str(), iter.socInfo.updateType, ret);
        }
        else {
            UPDATE_LOG_D("post soc: %s, updateType: %d, update continue task(%p) success, ret = %d(eContinue)",
                iter.socInfo.name.c_str(), iter.socInfo.updateType, task, ret);
        }
    }
    return 0;
}

uint8_t
UpdateAgent::Active()
{
    UPDATE_LOG_D("Active");

    SocManifest_t socManifests = ConfigManager::Instance().GetSocManifest();
    for (auto iter : socManifests.socs) {
        UpdateNTTaskSocInstaller* task = new UpdateNTTaskSocInstaller(nullptr, nullptr, iter, true);
        int32_t ret = UpdateAgent::Instance().post(task);
        if (eContinue != ret) {
            UPDATE_LOG_E("post update continue task fail, ret = %d", ret);
            return 2;
        }
        else {
            UPDATE_LOG_D("post update continue task(%p) success, ret = %d(eContinue)", task, ret);
        }
    }

    return 0;
}

uint8_t
UpdateAgent::RollBack()
{
    UPDATE_LOG_D("RollBack");
    return 0;
}

uint8_t
UpdateAgent::Cancel()
{
    UPDATE_LOG_D("Cancel");
    return 0;
}

void UpdateAgent::onInit()
{
    UPDATE_LOG_D("UpdateAgent onInit()");
    // register command task channels
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_SOC1);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_SOC2);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_MDC);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_LIDAR_FL);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_LIDAR_FR);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_SRR_FL);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_SRR_FR);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_SRR_RL);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_SRR_RR);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_LRR);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_USSC);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_IMU);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_CAN_FUNCTION);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_ETH_FUNCTION);
    getConfiguration()->configCommandChannel(OTA_TASK_CHANNEL_INTERFACE);

    // register eth and can channel command tasks
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_MDC, OTA_TASK_CHANNEL_MDC);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_LIDAR_FL, OTA_TASK_CHANNEL_LIDAR_FL);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_LIDAR_FR, OTA_TASK_CHANNEL_LIDAR_FR);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_SRR_FL, OTA_TASK_CHANNEL_SRR_FL);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_SRR_FL, OTA_TASK_CHANNEL_SRR_FL);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_SRR_FR, OTA_TASK_CHANNEL_SRR_FR);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_SRR_RL, OTA_TASK_CHANNEL_SRR_RL);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_SRR_RR, OTA_TASK_CHANNEL_SRR_RR);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_LRR, OTA_TASK_CHANNEL_LRR);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_USSC, OTA_TASK_CHANNEL_USSC);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_IMU, OTA_TASK_CHANNEL_IMU);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_CAN_FUNCTION, OTA_TASK_CHANNEL_CAN_FUNCTION);
    getConfiguration()->registerCommand(OTA_CTTASK_SEND_COMMAND_ETH_FUNCTION, OTA_TASK_CHANNEL_ETH_FUNCTION);
    getConfiguration()->registerCommand(OTA_CTTASK_INTERFACE_UPDATE, OTA_TASK_CHANNEL_INTERFACE);
    getConfiguration()->registerCommand(OTA_CTTASK_INTERFACE_ACTIVATE, OTA_TASK_CHANNEL_INTERFACE);
    getConfiguration()->registerCommand(OTA_CTTASK_INTERFACE_FINISH, OTA_TASK_CHANNEL_INTERFACE);
    getConfiguration()->registerCommand(OTA_CTTASK_INTERFACE_WAIT, OTA_TASK_CHANNEL_INTERFACE);
    // register timer tasks
    getConfiguration()->registerTimer(TIMER_INIT_DELAY);
    getConfiguration()->registerTimer(TIMER_COMMANDS_DELAY);
    getConfiguration()->registerTimer(TIMER_UPDATE_RETRY_DELAY);

    // register normal task, only once in queue
    getConfiguration()->registerOperation(OTA_NTTASK_INIT);
    getConfiguration()->registerOperation(OTA_NTTASK_RESET);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_MDC);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_IMU);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_LIDAR_FL);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_LIDAR_FR);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_LRR);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_MCU);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_SOC1);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_SOC2);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_SRR_FL);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_SRR_FR);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_SRR_RL);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_SRR_RR);
    getConfiguration()->registerOperation(OTA_NTTASK_INSTALLER_USSC);

    // register normal task, could be serval in queue, delete if reach the range
    STTaskConfig::OPERATION_CONFIG operaQueueConfig;
    operaQueueConfig.maxOperationCount = OTA_DUPLICATE_TASK_IN_QUEUE_MAX;
    operaQueueConfig.queueMethod = STTaskConfig::QUEUE_METHOD_EXCLUDE_EXECUTING | STTaskConfig::QUEUE_METHOD_DELETE_FRONT_IF_OVERFLOW;
    getConfiguration()->registerOperation(OTA_NTTASK_SEND_COMMANDS, operaQueueConfig);
    getConfiguration()->registerOperation(OTA_NTTASK_SECURITY_ACCESS, operaQueueConfig);
    getConfiguration()->registerOperation(OTA_NTTASK_TRANSFER_FILE, operaQueueConfig);
}

uint32_t UpdateAgent::onOperationStart(uint32_t operationId, STNormalTask* topTask)
{
    return eContinue;
}

void UpdateAgent::onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* task)
{
    if (nullptr != task) {
        switch (task->getOperationId()) {
        case OTA_NTTASK_INIT:
            {
                if (eOK == result) {
                    UPDATE_LOG_D("Task init completed success!");
                }
                else {
                    UPDATE_LOG_E("Task init completed failed!");
                }
            }
            break;
        case eOperation_HandleEvent:
            break;
        case OTA_NTTASK_SEND_COMMANDS:
            {
                if (eOK == result) {
                    UPDATE_LOG_D("Task send command completed success!");
                }
                else {
                    UPDATE_LOG_E("Task send command completed failed!");
                }
            }
            break;
        case OTA_NTTASK_INSTALLER_USSC:
            break;
        case OTA_NTTASK_INSTALLER_SRR_FL:
            break;
        case OTA_NTTASK_INSTALLER_SRR_FR:
            break;
        case OTA_NTTASK_INSTALLER_SRR_RL:
            break;
        case OTA_NTTASK_INSTALLER_SRR_RR:
            break;
        case OTA_NTTASK_INSTALLER_LIDAR_FL:
            break;
        case OTA_NTTASK_INSTALLER_LIDAR_FR:
            break;
        case OTA_NTTASK_INSTALLER_MDC:
            break;
        default:
            UPDATE_LOG_D("unknown top task, task operation=%d", task->getOperationId());
            return;
        }
    }
}

uint32_t UpdateAgent::onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
{
    return eContinue;
}

void UpdateAgent::onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask)
{
}

void UpdateAgent::onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
{
    if (nullptr == event) {
        UM_WARN << "event is nullptr";
        return;
    }

}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
