#include "update_manager/cmd_line_upgrade/sensors_uds_manager.h"
#include "update_manager/common/common_operation.h"
#include "update_manager/log/update_manager_logger.h"
#include "json/json.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/config/update_settings.h"

namespace hozon {
namespace netaos {
namespace update {

SensorsUdsManager* SensorsUdsManager::m_pInstance = nullptr;
std::mutex SensorsUdsManager::m_mtx;

#define SENSORS_CONFIG_FILE_ORIN      ("/app/runtime_service/update_manager/conf/sensors_config.json")
#define SENSORS_CONFIG_FILE_DEFAULT   ("/app/runtime_service/update_manager/conf/sensors_config.json")

SensorsUdsManager::SensorsUdsManager():
messageSuccess(false)
,isStopped(false)
{
}

SensorsUdsManager::~SensorsUdsManager()
{
}

SensorsUdsManager*
SensorsUdsManager::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new SensorsUdsManager();
        }
    }

    return m_pInstance;
}

void
SensorsUdsManager::Init()
{
    UM_INFO << "SensorsUdsManager::Init.";
    PraseJsonFile();
    UM_INFO << "SensorsUdsManager::Init Done.";
}

void
SensorsUdsManager::Deinit()
{
    UM_INFO << "SensorsUdsManager::Deinit.";
    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "SensorsUdsManager::Deinit Done.";
}

void 
SensorsUdsManager::Start()
{
    UM_INFO << "SensorsUdsManager::Start.";
    std::thread([this]() {
        UPDATE_LOG_D("SensorsUdsManager::Start");
        std::string lidar_version{};
        std::string srr_fl_version{};
        std::string srr_fr_version{};
        std::string srr_rl_version{};
        std::string srr_rr_version{};
        std::string curState = UpdateStateMachine::Instance()->GetCurrentState();
        if (GetVersion(SENSOR_LIDAR, lidar_version)) {
            OTAStore::Instance()->WriteDynamicSensorVersionData(SENSOR_LIDAR, lidar_version);
        }
        ReleaseConnection(SENSOR_LIDAR);
        if (GetVersion(SENSOR_SRR_FL, srr_fl_version)) {
            OTAStore::Instance()->WriteDynamicSensorVersionData(SENSOR_SRR_FL, srr_fl_version);
        }
        if (GetVersion(SENSOR_SRR_FR, srr_fr_version)) {
            OTAStore::Instance()->WriteDynamicSensorVersionData(SENSOR_SRR_FR, srr_fr_version);
        }
        if (GetVersion(SENSOR_SRR_RL, srr_rl_version)) {
            OTAStore::Instance()->WriteDynamicSensorVersionData(SENSOR_SRR_RL, srr_rl_version);
        }
        if (GetVersion(SENSOR_SRR_RR, srr_rr_version)) {
            OTAStore::Instance()->WriteDynamicSensorVersionData(SENSOR_SRR_RR, srr_rr_version);
        }
        ReleaseConnection(SENSOR_SRR_RR);
        UM_INFO << "LIDAR version is : " <<  lidar_version;
        UM_INFO << "SRR_FL version is : " <<  srr_fl_version;
        UM_INFO << "SRR_FR version is : " <<  srr_fr_version;
        UM_INFO << "SRR_RL version is : " <<  srr_rl_version;
        UM_INFO << "SRR_RR version is : " <<  srr_rr_version;
    }).detach();
    UM_INFO << "SensorsUdsManager::Start Done.";
}

void
SensorsUdsManager::Stop()
{
    UM_INFO << "SensorsUdsManager::Stop.";
    UM_DEBUG << "notify !";
    isStopped = true;
    msg_received_cv_.notify_all();
    UM_INFO << "SensorsUdsManager::Stop Done.";
}

bool 
SensorsUdsManager::PraseJsonFile(){

    UPDATE_LOG_D("SensorsUdsManager::PraseJsonFile");
    std::string jsonFile{};
#ifdef BUILD_FOR_ORIN
    jsonFile = SENSORS_CONFIG_FILE_ORIN;
#else
    jsonFile = SENSORS_CONFIG_FILE_DEFAULT;
#endif
    UPDATE_LOG_D("jsonFile path : %s", jsonFile.c_str());

    if (!PathExists(jsonFile)){
        UPDATE_LOG_W("file_list.json not Exists");
        return false;
    }

    Json::CharReaderBuilder reader;
    Json::Value root;
    std::string errs;
    std::ifstream ifs(jsonFile);
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if(!res)
    {
        ifs.close();
        UPDATE_LOG_E("Json parse error.");
        return false;
    }

    const Json::Value& sensorsInfo = root["sensorsInfo"];
    for (const auto& sensor : sensorsInfo) {
        SensorsDataInfo info{};
        info.sensorName = sensor["name"].asString();
        std::string logicAddrStr = sensor["logicAddr"].asString();
        // Convert hex string to uint16_t
        info.logicAddr = std::stoul(logicAddrStr, nullptr, 16);
        info.busType = sensor["busType"].asUInt();
        infos_.emplace_back(info);
    }
    ifs.close();
    return true;
}

bool 
SensorsUdsManager::GetVersion(const std::string& snesorName, std::string& snesorVersion)
{
    if (isStopped)
    {
        UM_ERROR << "SensorsUdsManager Stopped! Can not get Version.";
        snesorVersion = "";
        return false;
    }
    
    UM_DEBUG << "SensorsUdsManager::GetVersion, sensor name is : " << snesorName;
    if (infos_.empty())
    {
        UM_ERROR << "SensorsDataInfo is empty!";
        snesorVersion = "";
        return false;
    }
    
    auto it = infos_.begin();
    for(; it != infos_.end(); it++) {
        if (snesorName == it->sensorName) {
            break;
        }
    }
    if (it != infos_.end()) {
        auto res = SendRequest(*it, snesorVersion);
        if (res) {
            UM_DEBUG << "get version success, sensor name is : " << snesorName << " , version is : " << snesorVersion;
        } else {
            UM_ERROR << "get version failed, sensor name is : " << snesorName;
            snesorVersion = "";
            return false;
        }
    }
    else {
        UM_ERROR << "this sensor is not config in sensors_config.json, name is : " << snesorName;
        snesorVersion = "";
        return false;
    }
    return true;
}

bool
SensorsUdsManager::ReleaseConnection(const std::string& snesorName)
{
    UM_DEBUG << "SensorsUdsManager::ReleaseConnection, sensor name is : " << snesorName;
    if (infos_.empty())
    {
        UM_ERROR << "SensorsDataInfo is empty!";
        return false;
    }
    
    auto it = infos_.begin();
    for(; it != infos_.end(); it++) {
        if (snesorName == it->sensorName) {
            break;
        }
    }
    if (it != infos_.end()) {
        std::unique_ptr<uds_raw_data_req_t> rawDataReq = std::make_unique<uds_raw_data_req_t>();
        rawDataReq->sa = UpdateSettings::Instance().UmLogicAddr();
        rawDataReq->ta = it->logicAddr;
        rawDataReq->bus_type = 0x20;
        rawDataReq->data_vec = {};
        rawDataReq->data_len = 0;
        bool ret = DiagAgent::Instance()->SendUdsRawData(rawDataReq);
        if (ret) {
            UM_DEBUG << "release connect success, sensor name is : " << snesorName;
        } else {
            UM_ERROR << "release connect failed, sensor name is : " << snesorName;
            return false;
        }
    }
    else {
        UM_ERROR << "this sensor is not config in sensors_config.json, name is : " << snesorName;
        return false;
    }
    return true;
}

bool 
SensorsUdsManager::SendRequest(const SensorsDataInfo& info, std::string& version)
{
    UM_DEBUG << "SensorsUdsManager::SendRequest.";
    std::unique_lock<std::mutex> lock(mutex_);
    DiagAgent::Instance()->RegistReadVersionReceiveCallback([this, &version, info](const std::unique_ptr<uds_raw_data_resp_t>& uds_data) -> void {
        UPDATE_LOG_D("Uds receive callback updateType: %d, reqSa: %X, reqTa: %X, result: %d, uds data size: %ld.", uds_data->bus_type, uds_data->sa, uds_data->ta, uds_data->result, uds_data->data_vec.size());
        if (uds_data->data_vec[0] == 0x7F) {
            UM_ERROR << "receive 7F !";
            messageSuccess.store(false);
        }
        if (uds_data->result != 1) {
            UM_ERROR << "get response error ,result is not ok!";
            messageSuccess.store(false);
        }
        if (uds_data->sa != info.logicAddr) {
            UM_ERROR << "logic addr not match!";
            messageSuccess.store(false);
        }
        if (uds_data->data_vec[0] == 0x62 && uds_data->data_vec[1] == 0xF1 && uds_data->data_vec[2] == 0xC0) {
            version = std::string(reinterpret_cast<const char*>(&uds_data->data_vec[3]), uds_data->data_vec.size() - 3);
            messageSuccess.store(true);
        }
        UM_DEBUG << "notify !";
        msg_received_cv_.notify_one();
    });
    UM_INFO << "SensorsDataInfo.name is " << info.sensorName;
    UM_INFO << "SensorsDataInfo.busType is " << info.busType;
    UM_INFO << "SensorsDataInfo.logicAddr is " << info.logicAddr;

    std::unique_ptr<uds_raw_data_req_t> rawDataReq = std::make_unique<uds_raw_data_req_t>();
    rawDataReq->sa = UpdateSettings::Instance().UmLogicAddr();
    rawDataReq->ta = info.logicAddr;
    rawDataReq->bus_type = info.busType;
    rawDataReq->data_vec = {0x22, 0xF1, 0xC0};
    rawDataReq->data_len = 3;
    bool ret = DiagAgent::Instance()->SendUdsRawData(rawDataReq);
    if (!ret) {
        UM_ERROR << "send 22 F1 C0 error.";
        return false;
    }

    if (std::cv_status::timeout == msg_received_cv_.wait_for(lock, std::chrono::seconds(5))) {
        UM_ERROR << "Timeout: No response received within 5 seconds.";
        DiagAgent::Instance()->DeRegistReadVersionReceiveCallback();
        return false;
    } else {
        UM_DEBUG << "ReadVersionReceiveCallback success, wait checking received data.";
        if (!messageSuccess.load()) {
            UM_ERROR << "received but data not right.";
            DiagAgent::Instance()->DeRegistReadVersionReceiveCallback();
            return false;
        } else {
            UM_DEBUG << "22 F1 C0 get response success, version is : " << version;
            messageSuccess.store(false);
        }
    }
    DiagAgent::Instance()->DeRegistReadVersionReceiveCallback();
    return true;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
