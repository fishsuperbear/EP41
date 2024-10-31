#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/update_check/update_check.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/agent/update_agent.h"
#include "https/include/ota_download.h"
#include "update_manager/config/update_settings.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/common/common_operation.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/record/ota_result_store.h"
#include "update_manager/manager/uds_command_controller.h"
namespace hozon {
namespace netaos {
namespace update {

CmdUpgradeManager* CmdUpgradeManager::m_pInstance = nullptr;
std::mutex CmdUpgradeManager::m_mtx;


CmdUpgradeManager::CmdUpgradeManager():ecu_mode_(0)
{
    get_version_receiver_ = std::make_unique<DevmGetVersionMethodServer>();
    pre_check_receiver_ = std::make_unique<DevmPreCheckMethodServer>();
    progress_receiver_ = std::make_unique<DevmProgressMethodServer>();
    start_update_receiver_ = std::make_unique<DevmStartUpdateMethodServer>();
    update_status_receiver_ = std::make_unique<DevmUpdateStatusMethodServer>();
    start_finish_receiver_ = std::make_unique<DevmStartFinishMethodServer>();
    cur_partition_receiver_ = std::make_unique<DevmCurPartitionMethodServer>();
    switch_slot_receiver_ = std::make_unique<DevmSwitchSlotMethodServer>();
}

CmdUpgradeManager::~CmdUpgradeManager()
{
}

CmdUpgradeManager*
CmdUpgradeManager::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new CmdUpgradeManager();
        }
    }

    return m_pInstance;
}

void
CmdUpgradeManager::Init()
{
    UM_INFO << "CmdUpgradeManager::Init.";
    get_version_receiver_->Init();
    pre_check_receiver_->Init();
    progress_receiver_->Init();
    start_update_receiver_->Init();
    update_status_receiver_->Init();
    start_finish_receiver_->Init();
    cur_partition_receiver_->Init();
    switch_slot_receiver_->Init();
    UM_INFO << "CmdUpgradeManager::Init Done.";
}

void
CmdUpgradeManager::Deinit()
{
    UM_INFO << "CmdUpgradeManager::Deinit.";
    switch_slot_receiver_->DeInit();
    cur_partition_receiver_->DeInit();
    start_finish_receiver_->DeInit();
    update_status_receiver_->DeInit();
    start_update_receiver_->DeInit();
    progress_receiver_->DeInit();
    pre_check_receiver_->DeInit();
    get_version_receiver_->DeInit();

    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "CmdUpgradeManager::Deinit Done.";
}

void 
CmdUpgradeManager::UpdateStatusMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<update_status_resp_t>& resp)
{
    /**
     * 1. 返回UM的当前状态
    */
    resp->update_status = UpdateStateMachine::Instance()->GetCurrentState();
    resp->error_code = 0;
    resp->error_msg = "success";
}

void 
CmdUpgradeManager::PreCheckMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<precheck_resp_t>& resp)
{
    /**
     * 1. 检查车速， 设置结果
     * 2. 检查档位， 设置结果
     * 3. 检查空间大小， 设置结果
    */
    // TODO TODO
    if (UpdateCheck::Instance().UpdatePreConditionCheck() != 0) {
        UPDATE_LOG_D("precheck error!");
        resp->gear = true;
        resp->speed = true;
        resp->space = true;
        resp->error_code = 0;
        // resp->error_msg = "call UpdatePreConditionCheck error!";
        resp->error_msg = "success";
        return;
    }
    if (UpdateCheck::Instance().GetGear() == 0) {
        resp->gear = true;
    } else {
        resp->gear = true;
    }
    if (UpdateCheck::Instance().GetSpeed() != -1 && UpdateCheck::Instance().GetSpeed() <= 3) {
        resp->speed = true;
    } else {
        resp->speed = true;
    }
    if (UpdateCheck::Instance().GetSpaceEnough()) {
        resp->space = true;
    } else {
        resp->space = true;
    }
    resp->error_code = 0;
    resp->error_msg = "success";
}

void 
CmdUpgradeManager::ProgressMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<progress_resp_t>& resp)
{
    /**
     * 1. 检查UM状态，如果非升级状态，返回错误
     * 2. 否则查询总进度，返回
    */
    std::string curState = UpdateStateMachine::Instance()->GetCurrentState();
    uint32_t errorCode = 0;
    std::string errorMsg = "success";
    if (curState == "NORMAL_IDLE") {
        resp->progress = 0;
        errorCode = 1;
        errorMsg = "current state is idle, please enter update mode first.";
    } else if (curState == "OTA_UPDATE_FAILED") {
        resp->progress = 0;
        errorCode = UpdateStateMachine::Instance()->GetFailedCode();
        errorMsg = "update failed! error info is : " + UpdateStateMachine::Instance()->GetFailedCodeMsg() + " ,please reset and try again.";
    } else if (curState == "OTA_PRE_UPDATE") {
        resp->progress = 0;
    } else if (curState == "OTA_UPDATED") {
        resp->progress = 90;
    } else if (curState == "OTA_ACTIVED") {
        resp->progress = 100;
    } else {
        resp->progress = DiagAgent::Instance()->GetTotalProgress();
    }

    resp->error_code = errorCode;
    resp->error_msg = errorMsg;  
}

void 
CmdUpgradeManager::StartUpdateMethod(const std::shared_ptr<start_update_req_t>& req, std::shared_ptr<start_update_resp_t>& resp)
{
    /**
     * 1. 检查状态是否为Idle，其他状态不允许触发升级
     * 2. 判断是否需要先检查前置条件，若需要则执行preCheck
     * 3. 条件满足则进行升级，先进行解压，配置解析，再触发升级接口
    */
    if (UpdateStateMachine::Instance()->GetCurrentState() != "NORMAL_IDLE") {
        UPDATE_LOG_D("um state error!");
        resp->error_code = 2;
        resp->error_msg = "current state is not idle, please wait update task complete first.";
        return;
    }
    if (req->skip_version) {
        UPDATE_LOG_D("SetSameVersionUpdate succ!");
        UpdateSettings::Instance().SetSameVersionUpdate(true);
    }

    std::string realPath = req->package_path;
    if (req->package_path.rfind(".tar.lrz") == std::string::npos){
        std::string file_path_new = req->package_path + ".tar.lrz";
        std::rename(req->package_path.c_str(), file_path_new.c_str());
        realPath = file_path_new;
    }
    if (req->start_with_precheck == true) {
        std::shared_ptr<common_req_t> precheck_req = std::make_shared<common_req_t>();
        std::shared_ptr<precheck_resp_t> precheck_resp = std::make_shared<precheck_resp_t>();
        PreCheckMethod(precheck_req, precheck_resp);
        if (precheck_resp->gear == true && precheck_resp->speed == true && precheck_resp->space == true) {
            UPDATE_LOG_D("precheck condition passed !");
        } else {
            UPDATE_LOG_D("precheck condition failed !");
            resp->error_code = 1;
            resp->error_msg = "update pre check error !";
            return;
        }
    } else {
        // go on
    }
    SetCmdTriggerUpgradeFlag(true);
    OTARecoder::Instance().RestoreProgress();
    OTARecoder::Instance().RecordStart("TOTAL", 0);
    OtaResultStore::Instance()->ResetResult();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_PRE_UPDATE);
    // 先返回成功，后启动线程进行解压
    resp->error_code = 0;
    resp->error_msg = "success";
    updatReq_ = req;
    pkgPath_ = realPath;
    std::thread([this]() {
        UPDATE_LOG_D("UpdateManager switch into Update Mode.");
        UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_OTA);
        std::map<std::string, std::string> check_param{{"unzip_path", UpdateSettings::Instance().PathForUnzip()}};
        bool resVerify = hozon::netaos::https::OtaDownload::GetInstance().SetParam(check_param);
        if (resVerify) {
            PathCreate(UpdateSettings::Instance().PathForWork());
            PathCreate(UpdateSettings::Instance().PathForUnzip());
            PathCreate(UpdateSettings::Instance().PathForBinFiles());
            PathCreate(UpdateSettings::Instance().PathForUpdateTmp());
            PathCreate(UpdateSettings::Instance().PathForRecovery());
            PathCreate(UpdateSettings::Instance().PathForUpgrade());
            // realPath 需要暂存，用于激活阶段备份
            OTAStore::Instance()->WritePackageNameData(pkgPath_);
            resVerify = hozon::netaos::https::OtaDownload::GetInstance().Verify(pkgPath_);
            if (resVerify) {
                UPDATE_LOG_D("verify success!");
                ConfigManager::Instance().MountSensors();
                auto res = ConfigManager::Instance().ParseAllConfig();
                if (!res) {
                    UPDATE_LOG_D("Parse config error !");
                    UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::PARSE_CONFIG_FAILED);
                    return;
                } else {
                    // go on
                }
            }
            else {
                UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::VERIFY_FAILED);
                UPDATE_LOG_E("verify failed!");
                return;
            }
        }
        else {
            UPDATE_LOG_D("SetParam failed!");
            return;
        }
        if (updatReq_->ecu_mode == 0) {
            UM_DEBUG << "ECU update Mode is sensors & soc.";
        } else if (updatReq_->ecu_mode == 1) {
            UM_DEBUG << "ECU update Mode is soc only.";
            ConfigManager::Instance().UmountAndRemoveSensors();
            ConfigManager::Instance().ClearSensorUpdate();
        } else if (updatReq_->ecu_mode >= 2 && updatReq_->ecu_mode <= 6) {
            UM_DEBUG << "ECU update Mode is sensor only.";
            ConfigManager::Instance().ClearSocUpdate();
        } else {
            UPDATE_LOG_D("input ecu_mode error !");
            UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATE_FAILED, FailedCode::ECU_MODE_INVALIED);
            return;
        }
        SetEcuMode(updatReq_->ecu_mode);
        UpdateStateMachine::Instance()->SwitchState(State::OTA_UPDATING);
        UpdateAgent::Instance().Update();
    }).detach();
}

bool 
CmdUpgradeManager::IsCmdTriggerUpgrade()
{   bool flag = false;
    if(OTAStore::Instance()->ReadCmdFlagData(flag)) {
        cmd_upgrade_flag_ = flag;
        UM_INFO << "get Cfg CmdFlag is : " << cmd_upgrade_flag_;
    } else {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        if(GetVersionFormKey(CMD_FLAG_FILE, "ota/CmdFlag", flag)) {
            cmd_upgrade_flag_ = flag;
        }
    }
    return cmd_upgrade_flag_;
}

bool 
CmdUpgradeManager::SetCmdTriggerUpgradeFlag(const bool flag)
{
    OTAStore::Instance()->WriteCmdFlagData(flag);
    cmd_upgrade_flag_ = flag;
    return true;
}

bool 
CmdUpgradeManager::SetEcuMode(std::uint16_t mode)
{
    ecu_mode_ = mode;
    return true;
}

std::string 
CmdUpgradeManager::GetEcuMode()
{
    std::string ecu_ = "";
    switch (ecu_mode_)
    {
    case 0:
    case 1:
        ecu_ = "DEFAULT";
        break;
    case 2:
        ecu_ = "LIDAR";
        break;
    case 3:
        ecu_ = "SRR_FL";
        break;
    case 4:
        ecu_ = "SRR_FR";
        break;
    case 5:
        ecu_ = "SRR_RL";
        break;
    case 6:
        ecu_ = "SRR_RR";
        break;
    default:
        break;
    }
    return ecu_;
}

void 
CmdUpgradeManager::GetVersionMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<get_version_resp_t>& resp) 
{
    /**
     * 1. 获取大版本，失败返回错误
     * 2. 获取Soc版本，失败返回错误
     * 3. 获取所有sensor版本，失败返回错误
    */
    std::string major_version{};
    std::string soc_version{};
    std::string mcu_version{};
    std::string dsv_version{};
    std::string swt_version{};

    std::string lidar_version{};
    std::string srr_fl_version{};
    std::string srr_fr_version{};
    std::string srr_rl_version{};
    std::string srr_rr_version{};

    std::vector<uint8_t> ver{};
    if (OTAStore::Instance()->ReadECUVersionData(ver)) {
        major_version.assign(ver.begin(), ver.end());
    } else {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_DIDS_FILE, "F1C0", major_version);
    }
    if(!OTAStore::Instance()->ReadSocVersionData(soc_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, "SOC", soc_version);
    }

    if(!OTAStore::Instance()->ReadMcuVersionData(mcu_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, "MCU", mcu_version);
    }

    OTAAgent::Instance()->GetVersionInfo(dsv_version);
    dsv_version.erase(dsv_version.find_last_not_of('\0') + 1);

    if(!OTAStore::Instance()->ReadSwitchVersionData(swt_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, "SWT", swt_version);
    }

    if(!OTAStore::Instance()->ReadSensorVersionData(SENSOR_LIDAR, lidar_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, SENSOR_LIDAR, lidar_version);
    }
    if(!OTAStore::Instance()->ReadSensorVersionData(SENSOR_SRR_FL, srr_fl_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, SENSOR_SRR_FL, srr_fl_version);
    }
    if(!OTAStore::Instance()->ReadSensorVersionData(SENSOR_SRR_FR, srr_fr_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, SENSOR_SRR_FR, srr_fr_version);
    }
    if(!OTAStore::Instance()->ReadSensorVersionData(SENSOR_SRR_RL, srr_rl_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, SENSOR_SRR_RL, srr_rl_version);
    }
    if(!OTAStore::Instance()->ReadSensorVersionData(SENSOR_SRR_RR, srr_rr_version)) {
        UM_DEBUG << "Read CFG error, now read cfg json directly.";
        GetVersionFormKey(CFG_VERSION_FILE, SENSOR_SRR_RR, srr_rr_version);
    }

    resp->major_version = major_version;
    resp->soc_version = soc_version;
    resp->mcu_version = mcu_version;
    resp->dsv_version = dsv_version;
    resp->swt_version = swt_version;
    resp->sensor_version = {{SENSOR_LIDAR, lidar_version}, {SENSOR_SRR_FL, srr_fl_version}, {SENSOR_SRR_FR, srr_fr_version}, {SENSOR_SRR_RL, srr_rl_version}, {SENSOR_SRR_RR, srr_rr_version}};
    resp->error_code = 0;
    resp->error_msg = "success";
}

void 
CmdUpgradeManager::StartFinishMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<start_finish_resp_t>& resp)
{
    /**
     * 1. 状态非update_failed 或者 actived，均返回错误
    */
    std::string curState = UpdateStateMachine::Instance()->GetCurrentState();
    if (curState == "NORMAL_IDLE") {
        resp->error_code = 1;
        resp->error_msg = "current state is " + curState + " , cannot execute finish step! please reset first.";
    } else {
        UPDATE_LOG_D("UpdateManager switch into Normal Mode.");
        UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_NORMAL);
        UpdateStateMachine::Instance()->ForceSetState(State::NORMAL_IDLE);
        UPDATE_LOG_D("CMD finish, now do PostUpdateProcess.");
        UpdateStateMachine::Instance()->PostUpdateProcess();
        resp->error_code = 0;
        resp->error_msg = "success";
    }
}

void 
CmdUpgradeManager::PartitionMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<cur_pratition_resp_t>& resp)
{
    std::string cur{};
    auto res = OTAAgent::Instance()->GetCurrentSlot(cur);
    if (!res) {
        UM_ERROR << "GetCurrentSlot error.";
        resp->cur_partition = "";
        resp->error_code = 1;
        resp->error_msg = "get current partition error";
        return;
    }
    resp->cur_partition = cur;
    resp->error_code = 0;
    resp->error_msg = "success";
}

void 
CmdUpgradeManager::SwitchSlotMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<switch_slot_resp_t>& resp)
{
    auto res = OTAAgent::Instance()->SwitchSlot();
    if (!res) {
        UM_ERROR << "SwitchSlot error.";
        resp->error_code = 1;
        resp->error_msg = "switch slot error";
        return;
    }
    resp->error_code = 0;
    resp->error_msg = "success";
}

bool 
CmdUpgradeManager::GetVersionFormKey(const std::string& filePath, const std::string& key, std::string& version) 
{
    UM_DEBUG << "CmdUpgradeManager::GetVersionFormKey.";
    std::ifstream fileStream(filePath);
    if (!fileStream.is_open()) {
        UM_ERROR << "Error opening JSON file.";
        version = "";
        return false;
    }

    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    Json::parseFromStream(readerBuilder, fileStream, &root, nullptr);

    fileStream.close();

    const Json::Value& kvVec = root["kv_vec"];
    if (!kvVec.isArray()) {
        UM_ERROR << "Invalid JSON format. 'kv_vec' should be an array.";
        version = "";
        return false;
    }

    for (const auto& kv : kvVec) {
        if (kv["key"].asString() == key) {
            version = kv["value"]["string"].asString();
            UM_INFO << "get cfg key is : " << key << " ,value is : " << version; 
            return true;
        }
    }

    UM_ERROR << "Key '" << key << "' not found in JSON file.";
    version = "";
    return false;
}

bool 
CmdUpgradeManager::GetVersionFormKey(const std::string& filePath, const std::string& key, bool& flag) 
{
    UM_DEBUG << "CmdUpgradeManager::GetVersionFormKey.";
    std::ifstream fileStream(filePath);
    if (!fileStream.is_open()) {
        UM_ERROR << "Error opening JSON file.";
        return false;
    }

    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    Json::parseFromStream(readerBuilder, fileStream, &root, nullptr);

    fileStream.close();

    const Json::Value& kvVec = root["kv_vec"];
    if (!kvVec.isArray()) {
        UM_ERROR << "Invalid JSON format. 'kv_vec' should be an array.";
        return false;
    }

    for (const auto& kv : kvVec) {
        if (kv["key"].asString() == key) {
            flag = kv["value"]["string"].asBool();
            UM_INFO << "get cfg key is : " << key << " ,value is : " << flag; 
            return true;
        }
    }

    UM_ERROR << "Key '" << key << "' not found in JSON file.";
    return false;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
