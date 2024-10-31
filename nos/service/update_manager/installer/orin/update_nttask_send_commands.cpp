/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: can commands normal task
 */

#include "update_nttask_send_commands.h"
#include "update_cttask_command.h"
#include "update_tmtask_delay_timer.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/config/sensor_entity_manager.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/common/um_functions.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {


UpdateNTTaskSendCommands::UpdateNTTaskSendCommands(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
            const SensorInfo_t& sensorInfo, const UpdateCase_t& process, bool isTopTask)
    : NormalTaskBase(OTA_NTTASK_SEND_COMMANDS, pParent, pfnCallback, isTopTask)
    , m_sensorInfo(sensorInfo)
    , m_process(process)
{
}

UpdateNTTaskSendCommands::~UpdateNTTaskSendCommands()
{
}

TaskReqInfo& UpdateNTTaskSendCommands::GetReqInfo()
{
    return m_reqInfo;
}

TaskResInfo& UpdateNTTaskSendCommands::GetResInfo()
{
    return m_resInfo;
}

uint32_t UpdateNTTaskSendCommands::doAction()
{
    return StartToSendUdsCommand();
}

void UpdateNTTaskSendCommands::onCallbackAction(uint32_t result)
{
}

uint32_t UpdateNTTaskSendCommands::StartToSendUdsCommand()
{
    UPDATE_LOG_D("StartToSendUdsCommand!~");
    // data buffer to transfer

    m_reqInfo.reqUpdateType = m_sensorInfo.updateType;
    m_reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    m_reqInfo.reqTa = m_sensorInfo.logicalAddr;
    m_reqInfo.reqContent = m_process.transData;
    m_reqInfo.reqExpectContent = m_process.recvExpect;
    m_reqInfo.reqWaitTime = m_process.waitTime;
    if (m_process.transData.size() < m_process.transDataSize) {
        // need append transfer data, for example $2E F190/F198/F199
        if (m_process.transData[0] == 0x2E) {
            if (m_process.transData[1] == 0xF1 && m_process.transData[2] == 0x90) {
                // write 17 bytes vin
                std::vector<uint8_t> vin;
                std::vector<uint8_t> uds_req = { 0x2E, 0xF1, 0x90 };
                ConfigManager::Instance().GetVin(vin);
                uds_req.insert(uds_req.end(), vin.begin(), vin.end());
                m_reqInfo.reqContent = uds_req;
            }
            else if (m_process.transData[1] == 0xF1 && m_process.transData[2] == 0x98) {
                // write 10 bytes tester SN
                std::vector<uint8_t> sn;
                std::vector<uint8_t> uds_req = { 0x2E, 0xF1, 0x98 };
                ConfigManager::Instance().GetTesterSN(sn);
                uds_req.insert(uds_req.end(), sn.begin(), sn.end());
                m_reqInfo.reqContent = uds_req;
            }
            else if (m_process.transData[1] == 0xF1 && m_process.transData[2] == 0x99) {
                // write 4 bytes programming date
                std::vector<uint8_t> date;
                std::vector<uint8_t> uds_req = { 0x2E, 0xF1, 0x99 };
                ConfigManager::Instance().GetDate(date);
                uds_req.insert(uds_req.end(), date.begin(), date.end());
                m_reqInfo.reqContent = uds_req;
            }
        }
    }

    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                    CAST_TASK_CB(&UpdateNTTaskSendCommands::OnSendUdsCommandResult),
                                    m_reqInfo,
                                    m_resInfo);


    return post(task);
}

void UpdateNTTaskSendCommands::OnSendUdsCommandResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnSendUdsCommandResult result: %s!~", GetTaskResultString(result).c_str());
    if (N_OK == result) {
        UpdateCTTaskCommand* sftask = static_cast<UpdateCTTaskCommand*>(task);
        if (nullptr == sftask) {
            result = N_ERROR;
            onCallbackResult(result);
            return;
        }

        m_resInfo.resContent = sftask->GetResInfo().resContent;
        if (m_resInfo.resContent.size() > 3 && m_resInfo.resContent[0] == 0x62) {
            if (m_resInfo.resContent[1] == 0xF1 && m_resInfo.resContent[2] == 0x87) {
                std::string recvPartNumber = std::string(reinterpret_cast<const char*>(&m_resInfo.resContent[3]), m_resInfo.resContent.size() - 3);

                // 根据partNumber筛选与之对应的的entity
                auto res = SensorEntityManager::Instance()->ParseEntityByPartNum(m_sensorInfo.name, recvPartNumber, m_sensorInfo.entitys);
                if (!res) {
                    UM_DEBUG << "ParseEntityByPartNum error !";
                    result = N_ERROR;
                    onCallbackResult(result);
                    return;
                } 
                
                OTARecoder::Instance().UpdateSensorProgressVersion(m_sensorInfo.name, SensorEntityManager::Instance()->GetTargetVersion(m_sensorInfo.name));
                // 重新解析配置信息， 并更新
                auto parseSensor = ConfigManager::Instance().ParseSensorManifest();
                if (parseSensor != 0)
                {
                    UM_DEBUG << "ParseSensorManifest error !";
                    result = N_ERROR;
                    onCallbackResult(result);
                    return;
                }

                std::string rightPartNum = SensorEntityManager::Instance()->GetPartNumber(m_sensorInfo.name);
                // Read Part Number $62 F1 87
                bool match = true;
                if (m_resInfo.resContent.size() - 3 < rightPartNum.size()) {
                    UPDATE_LOG_D("partNumber size: %ld.", rightPartNum.size());
                    match = false;
                }
                for (uint8_t index = 0; index < rightPartNum.size(); ++index) {
                    if (rightPartNum[index] != m_resInfo.resContent[index + 3]) {
                        match = false;
                        break;
                    }
                }
                UPDATE_LOG_D("partNumber match: %d, expect : %s, recv: %s, size: %ld",
                    match, rightPartNum.c_str(), recvPartNumber.c_str(), m_resInfo.resContent.size());
                if (!match) {
                    result = N_ERROR;
                    onCallbackResult(result);
                    return;
                }
            }

            if (m_resInfo.resContent[1] == 0xF1 && m_resInfo.resContent[2] == 0x8A) {
                std::string rightSupplierCode = SensorEntityManager::Instance()->GetSupplierCode(m_sensorInfo.name);

                // Read Supplier Code $62 F1 8A
                bool match = true;
                if (m_resInfo.resContent.size() - 3 < rightSupplierCode.size()) {
                    match = false;
                }
                for (uint8_t index = 0; index < rightSupplierCode.size(); ++index) {
                    if (rightSupplierCode[index] != m_resInfo.resContent[index + 3]) {
                        match = false;
                        break;
                    }
                }
                std::string recvSupplierCode = std::string(reinterpret_cast<const char*>(&m_resInfo.resContent[3]), m_resInfo.resContent.size() - 3);
                UPDATE_LOG_D("supplierCode match: %d, expect: %s, recv: %s, size: %ld.",
                    match, rightSupplierCode.c_str(), recvSupplierCode.c_str(), m_resInfo.resContent.size());
                if (!match) {
                    result = N_ERROR;
                    onCallbackResult(result);
                    return;
                }
            }
        }
        if (m_process.delayTime > 0) {
            result = StartToWaitDelay();
        }
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

uint32_t UpdateNTTaskSendCommands::StartToWaitDelay()
{
    UpdateTMTaskDelayTimer* task = new UpdateTMTaskDelayTimer(TIMER_COMMANDS_DELAY,
                                    this,
                                    CAST_TASK_CB(&UpdateNTTaskSendCommands::OnWaitDelayResult),
                                    m_process.delayTime);
    return post(task);
}

void UpdateNTTaskSendCommands::OnWaitDelayResult(STTask *task, uint32_t result)
{
    onCallbackResult(N_OK);
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
