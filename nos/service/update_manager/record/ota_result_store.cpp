#include "update_manager/record/ota_result_store.h"
#include "update_manager/common/data_def.h"
#include "update_manager/common/common_operation.h"
#include "update_manager/log/update_manager_logger.h"
namespace hozon {
namespace netaos {
namespace update {

OtaResultStore* OtaResultStore::m_pInstance = nullptr;
std::mutex OtaResultStore::m_mtx;

OtaResultStore::OtaResultStore():
keys_{"ORIN", "LIDAR", "SRR_FL", "SRR_FR", "SRR_RL", "SRR_RR"},
validValues_{"None", "Succ", "Fail"}
{
}

OtaResultStore::~OtaResultStore()
{
}

OtaResultStore*
OtaResultStore::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new OtaResultStore();
        }
    }

    return m_pInstance;
}

void
OtaResultStore::Init()
{
    UM_INFO << "OtaResultStore::Init.";
    auto res = createFile(OTA_RESULT_FILE_PATH);
    if (res != 0) {
        UPDATE_LOG_E("CreateStateFile error, code is : %d", res);
    }
    UM_INFO << "OtaResultStore::Init Done.";
}

void
OtaResultStore::Deinit()
{
    UM_INFO << "OtaResultStore::Deinit.";
    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "OtaResultStore::Deinit Done.";
}

bool 
OtaResultStore::ResetResult()
{
    UM_DEBUG << "OtaResultStore::ResetResult";
    Json::Value root;
    for (const auto &key : keys_) {
        root[key] = "None";
    }
    writeToJsonFile(root);
    return true;
}

bool 
OtaResultStore::UpdateResult(const std::string& key, const std::string& value)
{
    std::unique_lock<std::mutex> lck(m_mtx);
    UM_DEBUG << "OtaResultStore::UpdateResult";
    if (!isValidKey(key) || !isValidValue(value)) {
        UM_ERROR << "key or value invalid!";
        return false;
    }

    Json::Value root = readFromJsonFile();
    root[key] = value;
    writeToJsonFile(root);
    UM_DEBUG << "OtaResultStore::UpdateResult success.";
    return true;
}

bool 
OtaResultStore::GetResultByKey(const std::string& key, std::string& value)
{
    UM_DEBUG << "OtaResultStore::GetResultByKey";
    if (!isValidKey(key)) {
        UM_ERROR << "key invalid!";
        return false;
    }

    Json::Value root = readFromJsonFile();
    UM_INFO << key << ": " << root[key].asString();
    if (!root[key].isNull()) {
        value = root[key].asString();
    } else {
        return false;
    }
    return true;
}

std::string 
OtaResultStore::To_string()
{
    UM_DEBUG << "OtaResultStore::To_string";
    Json::Value root = readFromJsonFile();
    Json::StreamWriterBuilder writer;
    return Json::writeString(writer, root);
}

void 
OtaResultStore::writeToJsonFile(const Json::Value &root) {
    UM_DEBUG << "OtaResultStore::writeToJsonFile";
    std::ofstream file(OTA_RESULT_FILE_PATH);
    if (file.is_open()) {
        Json::StreamWriterBuilder writer;
        file << Json::writeString(writer, root);
        file.close();
    } else {
        UM_ERROR << "Error opening file: " << OTA_RESULT_FILE_PATH;
    }
}

Json::Value 
OtaResultStore::readFromJsonFile() {
    UM_DEBUG << "OtaResultStore::readFromJsonFile";
    std::ifstream file(OTA_RESULT_FILE_PATH);
    Json::Value root;

    if (file.is_open()) {
        file >> root;
        file.close();
    } else {
        UM_ERROR << "Error opening file: " << OTA_RESULT_FILE_PATH;
    }

    return root;
}

bool 
OtaResultStore::isValidKey(const std::string &key) {
    return std::find(keys_.begin(), keys_.end(), key) != keys_.end();
}

bool 
OtaResultStore::isValidValue(const std::string &value) {
    return std::find(validValues_.begin(), validValues_.end(), value) != validValues_.end();
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
