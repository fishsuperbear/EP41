/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: ota diag agent module
 */
#include "update_manager/agent/diag_agent.h"
#include "update_manager/agent/update_agent.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/manager/uds_command_controller.h"

#include "json/json.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/update_check/update_check.h"
#include "update_manager/config/update_settings.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "https/include/ota_download.h"
#include "update_manager/record/ota_result_store.h"

namespace hozon {
namespace netaos {
namespace update {

DiagAgent* DiagAgent::m_pInstance = nullptr;
std::mutex DiagAgent::m_mtx;


DiagAgent::DiagAgent()
    : uds_raw_data_req_dispatcher_(nullptr),
      uds_raw_data_resp_receiver_(nullptr),
      uds_data_method_receiver_(nullptr),
      chassis_info_method_sender_(nullptr),
      progress_(0)
{
    uds_raw_data_req_dispatcher_ = std::make_unique<UdsRawDataReqDispatcher>();
    uds_raw_data_resp_receiver_ = std::make_unique<UdsRawDataRespReceiver>();
    uds_data_method_receiver_ = std::make_unique<UdsDataMethodReceiver>();
    update_status_method_receiver_ = std::make_unique<UpdateStatusMethodReceiver>();
    chassis_info_method_sender_ = std::make_unique<ChassisInfoMethodSender>();
    uds_cur_session_receiver_ = std::make_unique<UdCurrentSessionReceiver>();
}

DiagAgent::~DiagAgent()
{
}

DiagAgent*
DiagAgent::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new DiagAgent();
        }
    }

    return m_pInstance;
}

void
DiagAgent::Init()
{
    UM_INFO << "DiagAgent::Init.";
    uds_raw_data_req_dispatcher_->Init();
    uds_raw_data_resp_receiver_->Init();
    uds_data_method_receiver_->Init();
    update_status_method_receiver_->Init();
    chassis_info_method_sender_->Init();
    uds_cur_session_receiver_->Init();

    uds_raw_data_resp_receiver_->RegistUdsRawDataReceiveCallback([this](const std::unique_ptr<uds_raw_data_resp_t>& uds_data) -> void {

            uds_rawdata_receive_callback_(uds_data);
        }
    );

    uds_raw_data_resp_receiver_->RegistReadVersionReceiveCallback([this](const std::unique_ptr<uds_raw_data_resp_t>& uds_data) -> void {
            if (read_version_receive_callback_ != nullptr) {
                read_version_receive_callback_(uds_data);
            }
        }
    );
    UM_INFO << "DiagAgent::Init Done.";
}

void
DiagAgent::Deinit()
{
    UM_INFO << "DiagAgent::Deinit.";
    uds_cur_session_receiver_->DeInit();
    uds_data_method_receiver_->DeInit();
    uds_raw_data_resp_receiver_->DeInit();
    uds_raw_data_req_dispatcher_->Deinit();
    update_status_method_receiver_->DeInit();
    chassis_info_method_sender_->DeInit();

    uds_data_method_receiver_ = nullptr;
    uds_raw_data_resp_receiver_ = nullptr;
    uds_raw_data_req_dispatcher_ = nullptr;
    update_status_method_receiver_ = nullptr;
    chassis_info_method_sender_ = nullptr;
    uds_cur_session_receiver_ = nullptr;

    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "DiagAgent::Deinit Done.";
}

void
DiagAgent::RegistUdsRawDataReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback)
{
    uds_rawdata_receive_callback_ = uds_rawdata_receive_callback;
}

void
DiagAgent::RegistReadVersionReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> read_version_receive_callback)
{
    read_version_receive_callback_ = read_version_receive_callback;
}

void 
DiagAgent::DeRegistReadVersionReceiveCallback()
{
    read_version_receive_callback_ = nullptr;
}

bool 
DiagAgent::SendChassisInfo(std::unique_ptr<chassis_info_t>& output_info)
{
    return chassis_info_method_sender_->ChassisMethodSend(output_info);
}

bool
DiagAgent::SendUdsRawData(const std::unique_ptr<uds_raw_data_req_t>& uds_data)
{
    bool bRet(false);
    UdsRawDataReqEvent udsRawDataReqEvent;

    udsRawDataReqEvent.ta = uds_data->ta;
    udsRawDataReqEvent.sa = uds_data->sa;
    udsRawDataReqEvent.bus_type = uds_data->bus_type;
    udsRawDataReqEvent.data_vec = uds_data->data_vec;

    if (uds_raw_data_req_dispatcher_) {
        uds_raw_data_req_dispatcher_->Send(udsRawDataReqEvent);
        bRet = true;
    }
    else {
        UPDATE_LOG_E("DiagAgent::SendUdsRawData uds_raw_data_req_dispatcher_ used without init");
        bRet = false;
    }

    return bRet;
}

void
DiagAgent::ReceiveUdsData(const std::shared_ptr<uds_data_req_t>& uds_data_req,
                             std::shared_ptr<uds_data_req_t>& uds_data_resp)
{
    UdsDataReceiveCallback(uds_data_req, uds_data_resp);
}

uint8_t
DiagAgent::UdsDataReceiveCallback(const std::shared_ptr<uds_data_req_t>& uds_data_req,
                                      std::shared_ptr<uds_data_req_t>& uds_data_resp)
{
    UPDATE_LOG_D("received method sid: %2X, subid:%2X, uds size: %ld, data: [%s].",
        uds_data_req->sid, uds_data_req->subid, uds_data_req->data_vec.size(),
        (UM_UINT8_VEC_TO_HEX_STRING(uds_data_req->data_vec)).c_str());
    uint8_t resCode = 0;
    switch (uds_data_req->sid) {
    case UDS_SID_22:
    {
        if (uds_data_req->data_len < 2) {
            UPDATE_LOG_E("received uds method req data data len exeption!");
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_NACK;
            uds_data_resp->data_len = 1;
            uds_data_resp->data_vec = {0x13};
            return 1;
        }

        if (uds_data_req->data_vec[0] == 0x01 && uds_data_req->data_vec[1] == 0x07) {
            DidsUpdateProgressDisplayResponse(uds_data_req, uds_data_resp);
        }

        if (uds_data_req->data_vec[0] == 0xF1 && uds_data_req->data_vec[1] == 0xC0) {

            std::vector<uint8_t> version{};
            OTAStore::Instance()->ReadECUVersionData(version);
            
            std::string curVersion(version.begin(), version.end());
            UPDATE_LOG_D("received UDS $22 F1 C0 version: %s ", curVersion.c_str());
            curVersion = ("" != curVersion) ? curVersion : "00.00.00";

            std::vector<uint8_t> data = {0xF1, 0xC0 };
            data.insert(data.end(), curVersion.begin(), curVersion.end());
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
            uds_data_resp->data_len = data.size();
            uds_data_resp->data_vec = data;

        }
        if (uds_data_req->data_vec[0] == 0x02 && uds_data_req->data_vec[1] == 0x88) {
            UM_DEBUG << "received uds method [22 02 88].";
            std::string otaResult = OtaResultStore::Instance()->To_string();
            UM_INFO << "Json otaResult : " << otaResult;
            std::vector<uint8_t> data = {0x02, 0x88};
            data.insert(data.end(), otaResult.begin(), otaResult.end());
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
            uds_data_resp->data_len = data.size();
            uds_data_resp->data_vec = data;
        }
        break;
    }
    case UDS_SID_2E:
    {
        UPDATE_LOG_D("uds data size: %ld, data: %s.", uds_data_req->data_vec.size(), (UM_UINT8_VEC_TO_HEX_STRING(uds_data_req->data_vec)).c_str());
        if (uds_data_req->data_vec[0] == 0xF1 && uds_data_req->data_vec[1] == 0x98) {
            std::vector<uint8_t> version {uds_data_req->data_vec.end() - 10, uds_data_req->data_vec.end()};
            OTAStore::Instance()->WriteTesterSNData(version);
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;

            std::string cmd{"2E F1 98"};
            if (UpdateStateMachine::Instance()->GetCurrentState() == "NORMAL_IDLE") {
                    uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                    uds_data_resp->data_len = 2;
                    uds_data_resp->data_vec = {0xF1, 0x98};
            } else {
                if (!UdsCommandController::Instance()->ProcessCommand(cmd))
                {
                    UPDATE_LOG_E("received uds method req uds command sequence Not Correct!");
                    uds_data_resp->resp_ack = UDS_METHOD_RES_NACK;
                    uds_data_resp->data_len = 1;
                    uds_data_resp->data_vec = {0x22};
                } else {
                    uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                    uds_data_resp->data_len = 2;
                    uds_data_resp->data_vec = {0xF1, 0x98};
                }
            }
        } else if (uds_data_req->data_vec[0] == 0xF1 && uds_data_req->data_vec[1] == 0x99) {
            std::vector<uint8_t> date {uds_data_req->data_vec.end() - 4, uds_data_req->data_vec.end()};
            OTAStore::Instance()->WriteProgrammingDateData(date);
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            std::string cmd{"2E F1 99"};
            if (UpdateStateMachine::Instance()->GetCurrentState() == "NORMAL_IDLE") {
                    uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                    uds_data_resp->data_len = 2;
                    uds_data_resp->data_vec = {0xF1, 0x99};
            } else {
                if (!UdsCommandController::Instance()->ProcessCommand(cmd))
                {
                    UPDATE_LOG_E("received uds method req uds command sequence Not Correct!");
                    uds_data_resp->resp_ack = UDS_METHOD_RES_NACK;
                    uds_data_resp->data_len = 1;
                    uds_data_resp->data_vec = {0x22};
                } else {
                    uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                    uds_data_resp->data_len = 2;
                    uds_data_resp->data_vec = {0xF1, 0x99};
                }
            }
        }

        break;
    }
    case UDS_SID_31:
    {
        if (uds_data_req->data_len < 2) {
            UPDATE_LOG_E("received uds method req data data len exeption!");
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_NACK;
            uds_data_resp->data_len = 1;
            uds_data_resp->data_vec = {0x13};
            return 1;
        }
        UPDATE_LOG_D("uds data size: %ld, data: %s.", uds_data_req->data_vec.size(), (UM_UINT8_VEC_TO_HEX_STRING(uds_data_req->data_vec)).c_str());
        if (uds_data_req->data_vec[0] == 0x02 && uds_data_req->data_vec[1] == 0x01) {
            OTARecoder::Instance().RecordStepStart("TOTAL", "Check File Existed", progress_);
            std::string fileName = std::string(reinterpret_cast<const char*>(&(uds_data_req->data_vec[2])), uds_data_req->data_vec.size() - 2);
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            if (0 == access(fileName.c_str(), F_OK)) {
                UPDATE_LOG_D("file: %s is existed.", fileName.c_str());
                uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                uds_data_resp->data_len = 3;
                uds_data_resp->data_vec = {0x02, 0x01, 0x00};
            }
            else {
                UPDATE_LOG_D("file: %s is not existed!", fileName.c_str());
                uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                uds_data_resp->data_len = 3;
                uds_data_resp->data_vec = {0x02, 0x01, 0x01};
            }
            uint32_t result = (0 == resCode) ? N_OK : N_ERROR;
            OTARecoder::Instance().RecordStepFinish("TOTAL", "Check File Existed", result, progress_);
        }
        else if (uds_data_req->data_vec[0] == 0x02 && uds_data_req->data_vec[1] == 0x03) {
            UPDATE_LOG_D("UpdateManager exec update pre check.");
            OtaResultStore::Instance()->ResetResult();
            progress_ = 0;
            OTARecoder::Instance().RestoreProgress();
            OTARecoder::Instance().RecordStart("TOTAL", progress_);
            OTARecoder::Instance().RecordStepStart("TOTAL", "PreCondition Check", progress_);
            resCode = UpdateCheck::Instance().UpdatePreConditionCheck();
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            if (0 == resCode) {
                uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                uds_data_resp->data_len = 3;
                uds_data_resp->data_vec = {0x02, 0x03, 0x02};
            }
            else {
                uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
                uds_data_resp->data_len = 3;
                uds_data_resp->data_vec = {0x02, 0x03, 0x05};
            }
            uint32_t result = (0 == resCode) ? N_OK : N_ERROR;
            OTARecoder::Instance().RecordStepFinish("TOTAL", "PreCondition Check", result, progress_);
        }
        else if (uds_data_req->data_vec[0] == 0xff && uds_data_req->data_vec[1] == 0x01) {
            UPDATE_LOG_D("UpdateManager update precheck, exec update file md5 check.");
            OTARecoder::Instance().RecordStepStart("TOTAL", "Check Dependency", progress_);
            std::vector<uint8_t> md5_vec {uds_data_req->data_vec.end() - 16, uds_data_req->data_vec.end()};
            std::vector<uint8_t> file_name_vec {uds_data_req->data_vec.begin() + 2, uds_data_req->data_vec.end() - 17};
            std::string file_name(reinterpret_cast<char*>(file_name_vec.data()), file_name_vec.size());
            UM_DEBUG << "file name is : " << file_name;
            std::string realPath = file_name;
            if (file_name.rfind(".tar.lrz") == std::string::npos){
                std::string file_path_new = file_name + ".tar.lrz";
                std::rename(file_name.c_str(), file_path_new.c_str());
                realPath = file_path_new;
            }
            OTAStore::Instance()->WritePackageNameData(realPath);
            bool resVerify{false};
            std::string md5 = UM_UINT8_VEC_TO_HEX_STRING(md5_vec);
            md5.erase(std::remove(md5.begin(), md5.end(), ' '), md5.end());
            UPDATE_LOG_D("unziPath = %s", UpdateSettings::Instance().PathForUnzip().c_str());
            UPDATE_LOG_D("zip_md5 = %s", md5.c_str());
            std::map<std::string, std::string> check_param{
                {"unzip_path", UpdateSettings::Instance().PathForUnzip()},
                {"zip_md5", md5}
            };
            // std::unique_ptr<hozon::netaos::https::OtaDownload> dc_ptr = std::make_unique<hozon::netaos::https::OtaDownload>();

            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
            uds_data_resp->data_len = 3;

            resVerify = hozon::netaos::https::OtaDownload::GetInstance().SetParam(check_param);
            if (resVerify) {
                PathCreate(UpdateSettings::Instance().PathForWork());
                PathCreate(UpdateSettings::Instance().PathForUnzip());
                PathCreate(UpdateSettings::Instance().PathForBinFiles());
                PathCreate(UpdateSettings::Instance().PathForUpdateTmp());
                PathCreate(UpdateSettings::Instance().PathForRecovery());
                PathCreate(UpdateSettings::Instance().PathForUpgrade());

                resVerify = hozon::netaos::https::OtaDownload::GetInstance().Verify(realPath);
                if (resVerify) {
                    UPDATE_LOG_D("Verify success!");
                    ConfigManager::Instance().MountSensors();
                    auto res = ConfigManager::Instance().ParseAllConfig();
                    if (!res) {
                        UPDATE_LOG_D("Parse config error !");
                        UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::PARSE_CONFIG_FAILED);
                        uds_data_resp->data_vec = {0xff, 0x01, 0x05};
                    } else {
                        uds_data_resp->data_vec = {0xff, 0x01, 0x02};
                    }
                }
                else {
                    UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::VERIFY_FAILED);
                    UPDATE_LOG_D("Verify failed!");
                    uds_data_resp->data_vec = {0xff, 0x01, 0x05};
                }
            }
            else {
                UPDATE_LOG_D("SetParam failed!");
                uds_data_resp->data_vec = {0xff, 0x01, 0x05};
            }
            uint32_t result = (resVerify) ? N_OK : N_ERROR;
            OTARecoder::Instance().RecordStepFinish("TOTAL", "Check Dependency", result, progress_);
        }
        else if (uds_data_req->data_vec[0] == 0x02 && uds_data_req->data_vec[1] == 0x05) {
            if (!OTARecoder::Instance().IsUpdateCompleted()) {
                UPDATE_LOG_E("received uds method req data conditions Not Correct!");
                uds_data_resp->meta_info = uds_data_req->meta_info;
                uds_data_resp->sid = uds_data_req->sid;
                uds_data_resp->subid = uds_data_req->subid;
                uds_data_resp->resp_ack = UDS_METHOD_RES_NACK;
                uds_data_resp->data_len = 1;
                uds_data_resp->data_vec = {0x22};
                return 1;
            }
            UPDATE_LOG_D("UpdateManager exec start install update package.");
            OTARecoder::Instance().RecordStepStart("TOTAL", "Update Install", progress_);
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
            uds_data_resp->data_len = 3;

            auto switchRes = UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATING);
            if (!switchRes) {
                uds_data_resp->data_vec = {0x02, 0x05, 0x05};
            } else {
                uds_data_resp->data_vec = {0x02, 0x05, 0x02};
                UpdateAgent::Instance().Update();
            }
            uint32_t result = (true == switchRes) ? N_OK : N_ERROR;
            OTARecoder::Instance().RecordStepFinish("TOTAL", "Update Install", result, progress_);
        }
        else if (uds_data_req->data_vec[0] == 0x02 && uds_data_req->data_vec[1] == 0x06) {
            UPDATE_LOG_D("UpdateManager exec change partition and recovery update package.");
            OTARecoder::Instance().SetActivateProcess();
            OTARecoder::Instance().RecordStart("TOTAL", progress_);
            OTARecoder::Instance().RecordStepStart("TOTAL", "Activate Change Partition", progress_);
            // Soc版本一致的case，无须触发切面
            if (!UdsCommandController::Instance()->IsSocVersionSame()) {
                UPDATE_LOG_D("Call OTAAgent SwitchSlot");
                OTAAgent::Instance()->SwitchSlot();
            }
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
            uds_data_resp->data_len = 3;
            uds_data_resp->data_vec = {0x02, 0x06, 0x02};
            OTARecoder::Instance().RecordStepFinish("TOTAL", "Activate Change Partition", N_OK, progress_);
        }
        break;
    }
    case UDS_SID_38:
    {
        break;
    }
    case UDS_SID_11:
    {   
        if (!OTARecoder::Instance().IsUpdateCompleted()) {
            UPDATE_LOG_E("received uds method req data conditions Not Correct!");
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_NACK;
            uds_data_resp->data_len = 1;
            uds_data_resp->data_vec = {0x22};
            return 1;
        }

        UPDATE_LOG_D("UpdateManager exec restart onbord.");
        uds_data_resp->meta_info = uds_data_req->meta_info;
        uds_data_resp->sid = uds_data_req->sid;
        uds_data_resp->subid = uds_data_req->subid;
        uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;


        #ifdef BUILD_FOR_MDC
            {
                uint8_t updateProgress = 0;
                uint8_t activeProgress = 0;
                std::string updateMessage;
                std::string activeMessage;
                OTAAgent::Instance()->GetUpdateProgress(updateProgress, updateMessage);
                OTAAgent::Instance()->GetActivationProgress(activeProgress, activeMessage);
                UPDATE_LOG_D("SoC update progress: %d, message: %s, activate progress: %d, message: %s.",
                    updateProgress, updateMessage.c_str(), activeProgress, activeMessage.c_str());
                OTARecoder::Instance().RecordStepStart("TOTAL", "Activate Reset", progress_);

                if ((updateProgress == 100 && activeProgress < 100) && ConfigManager::Instance().IsSocUpdate()) {
                    // need do update activate.
                    resCode = UpdateAgent::Instance().Active();
                }
                else {
                    resCode = system("mdc-dbg sm reset hard 1");
                }
                uint32_t result = (0 == resCode) ? N_OK : N_ERROR;
                OTARecoder::Instance().RecordStepFinish("TOTAL", "Activate Reset", result, progress_);
            }
        #else
            {
                // 非升级场景，调用1101，单板复位
                if (UpdateStateMachine::Instance()->GetCurrentState() == "NORMAL_IDLE") {
                    std::thread([this]() {
                        UM_DEBUG << "sync && reboot";
                        SystemSync();
                        OTAAgent::Instance()->HardReboot();
                    }).detach();
                } else {
                    std::string targetVer{};
                    ConfigManager::Instance().GetMajorVersion(targetVer);
                    std::vector<uint8_t> version{targetVer.begin(), targetVer.end()};
                    OTAStore::Instance()->WriteECUVersionData(version);
                    OTAStore::Instance()->WriteECUSWData(version);
                    OTARecoder::Instance().RecordStepStart("TOTAL", "Activate Reset", progress_);
                    std::thread([this]() {
                        UM_DEBUG << "sync && reboot";
                        SystemSync();
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        SystemSync();
                        if (UdsCommandController::Instance()->IsSocVersionSame()) {
                            OTAAgent::Instance()->HardReboot();
                        } else {
                            OTAAgent::Instance()->Reboot();
                        }
                    }).detach();
                }
            }
        #endif
        break;
    }
    case UDS_SID_28:
    {
        if (uds_data_req->data_len != 1) {
            UPDATE_LOG_E("received uds method req data data len exeption!");
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_NACK;
            uds_data_resp->data_len = 1;
            uds_data_resp->data_vec = {0x13};
            return 1;
        }
        
        if ( (uds_data_req->subid & 0x7F) == 0x03){
            // TODO DisableRxAndTx
            UPDATE_LOG_D("UpdateManager DisableRxAndTx.");
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
            UPDATE_LOG_D("[28 03] received, now do clear work path.");
	        PathClear(UpdateSettings::Instance().PathForUpgrade());
            UPDATE_LOG_D("UpdateManager switch into Update Mode.");
            UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_OTA);
            UpdateStateMachine::Instance()->SwitchState(State::OTA_PRE_UPDATE);
        }

        if ((uds_data_req->subid & 0x7F) == 0x00) {
            // TODO EnableRxAndTx
            UPDATE_LOG_D("UpdateManager EnableRxAndTx.");
            uds_data_resp->meta_info = uds_data_req->meta_info;
            uds_data_resp->sid = uds_data_req->sid;
            uds_data_resp->subid = uds_data_req->subid;
            uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;

            UPDATE_LOG_D("UpdateManager switch into Normal Mode.");
            
            UpdateStateMachine::Instance()->SwitchState(State::NORMAL_IDLE);
            UPDATE_LOG_D("UpdateManager switch into Normal Mode.");
            UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_NORMAL);
            UPDATE_LOG_D("[28 00] received, now do PostUpdateProcess.");
            UpdateStateMachine::Instance()->PostUpdateProcess();
        }
        break;
    }
    default:
        break;
    }

    return resCode;
}

int32_t 
DiagAgent::DidsUpdateProgressDisplayResponse(const std::shared_ptr<uds_data_req_t>& uds_data_req,
                                      std::shared_ptr<uds_data_req_t>& uds_data_resp)
{
    uint8_t result = 0x02;  // 0x00: success, 0x01: failed, 0x02: in progress.
    uint8_t totalProgress = 0;
    uint8_t sensorProgress = OTARecoder::Instance().GetSensorTotalProgress();
    uint8_t socProgress = OTARecoder::Instance().GetSocTotalProgress();
    bool sensorCompleted = OTARecoder::Instance().IsSensorUpdateCompleted();
    bool socCompleted = OTARecoder::Instance().IsSocUpdateCompleted();
    UM_INFO << "progress is : " << progress_;
    UM_INFO << "sensorProgress is : " << sensorProgress;
    UM_INFO << "socProgress is : " << socProgress;
    UM_INFO << "sensorCompleted is : " << sensorCompleted;
    UM_INFO << "socCompleted is : " << socCompleted;

    if (OTARecoder::Instance().IsUpdatingProcess()) {
        // in update process, update progress up to 90%
        if (ConfigManager::Instance().IsSensorUpdate() && ConfigManager::Instance().IsSocUpdate()) {
            // sensor and soc both in update progress.
            UPDATE_LOG_D("Both Sensor and SoC update, current Sensor progress: %d, completed: %d,  SoC Progress: %d, completed: %d",
                sensorProgress, sensorCompleted, socProgress, socCompleted);
            // sensor 升级失败，不影响整体升级进度
            if (sensorCompleted) {
                sensorProgress = 100;
            }
            totalProgress = (sensorProgress/10 + socProgress*9/10)*9/10;
            progress_ = (totalProgress > progress_) ? totalProgress : progress_;
            result = (socProgress == 100 && socCompleted) ? 0x00
                : (socProgress < 100 && socCompleted) ? 0x01
                : (sensorProgress < 100 && sensorCompleted) ? 0x01
                : 0x02;
        }
        else if (ConfigManager::Instance().IsSensorUpdate()) {
            // only has sensor update.
            UPDATE_LOG_D("Only Sensor update, current Sensor Progress: %d.", sensorProgress);

            totalProgress = sensorProgress*9/10;
            progress_ = (totalProgress > progress_) ? totalProgress : progress_;
            result = (sensorProgress == 100) ? 0x00
                : (sensorProgress < 100 && sensorCompleted) ? 0x01
                : 0x02;
        }
        else if (ConfigManager::Instance().IsSocUpdate()) {
            // only has soc update, soc update only for 90%, activate for another 10%
            UPDATE_LOG_D("Only SoC update, current SocProgress : %d", socProgress);

            totalProgress = socProgress*9/10;
            progress_ = (totalProgress > progress_) ? totalProgress : progress_;
            result = (socProgress == 100 && socCompleted) ? 0x00
                : (socProgress < 100 && socCompleted) ? 0x01
                : 0x02;
        }
        else {
            // neither sensor nor soc in update.
            UPDATE_LOG_D("current neither sensor nor soc in update, maybe in unzip Ota_Package or Parse Xml error ~");
            if (UpdateStateMachine::Instance()->GetPreState() == "OTA_UPDATE_FAILED") {
                progress_ = 0;
                result = 1;
            }
        }
    }
    else {
        // int activate process, update for 90%, activate for 10% whether there's sensor update
        if (!OTARecoder::Instance().IsSocUpdateProcess()) {
            UPDATE_LOG_D("no SoC in activate process, activate completed!~");
            result = 0x00;
            progress_ = 100;
        }
        else {
            UPDATE_LOG_D("SoC activate, activate progress: %d, completed: %d.", socProgress, socCompleted);
            totalProgress = 90 + socProgress/10;
            progress_ = (totalProgress > progress_) ? totalProgress : progress_;
            result = (socProgress == 100 && socCompleted) ? 0x00
                : (socProgress < 100 && socCompleted) ? 0x01
                : 0x02;
        }
    }

    UPDATE_LOG_D("update result: %d, progress: %d.", result, progress_);
    uds_data_resp->meta_info = uds_data_req->meta_info;
    uds_data_resp->sid = uds_data_req->sid;
    uds_data_resp->subid = uds_data_req->subid;
    uds_data_resp->resp_ack = UDS_METHOD_RES_ACK;
    uds_data_resp->data_len = 3;
    uds_data_resp->data_vec = {0x01, 0x07, result};

    if (result == 0x00 || result == 0x01) {
        uint32_t res = (0x00 == result) ? N_OK : N_ERROR;
        OTARecoder::Instance().RecordFinish("TOTAL", res, progress_);
    }
    return result;
}

uint8_t 
DiagAgent::GetTotalProgress()
{
    // 满足cmd升级需求，需要主动触发22 01 07 进行进度更新，否则拿不到进度
    std::shared_ptr<uds_data_req_t> uds_data_req = std::make_shared<uds_data_req_t>();
    std::shared_ptr<uds_data_req_t> uds_data_resp = std::make_shared<uds_data_req_t>();
    DidsUpdateProgressDisplayResponse(uds_data_req, uds_data_resp);
    UPDATE_LOG_D("getTotalProgress progress is : %d", progress_);
    return progress_;
}

bool 
DiagAgent::ResetTotalProgress()
{
    UPDATE_LOG_D("reset total progress !");
    progress_ = 0;
    return true;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
