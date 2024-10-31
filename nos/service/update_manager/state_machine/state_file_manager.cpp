#include "update_manager/state_machine/state_file_manager.h"
#include "update_manager/common/common_operation.h"
#include "update_manager/common/data_def.h"
#include "update_manager/config/update_settings.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

StateFileManager* StateFileManager::m_pInstance = nullptr;
std::mutex StateFileManager::m_mtx;

StateFileManager::StateFileManager()
{
}

StateFileManager::~StateFileManager()
{
}

StateFileManager*
StateFileManager::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new StateFileManager();
        }
    }

    return m_pInstance;
}

void
StateFileManager::Init()
{
    UM_INFO << "StateFileManager::Init.";
    UM_INFO << "StateFileManager::Init Done.";
}

void
StateFileManager::Deinit()
{
    UM_INFO << "StateFileManager::Deinit.";

    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "StateFileManager::Deinit Done.";
}
bool 
StateFileManager::CreateStateFile()
{
    auto res = createFile(STATE_FILE_PATH  + UM_STATE_FILE);
    if (res != 0) {
        UPDATE_LOG_E("CreateStateFile error, code is : %d", res);
        return false;
    }
    return true;
}

bool 
StateFileManager::UpdateStateFile(const std::string& stateContent)
{
    auto res = writeToFile(STATE_FILE_PATH  + UM_STATE_FILE, stateContent);
    if (res != 0) {
        UPDATE_LOG_E("UpdateStateFile error, code is : %d", res);
        return false;
    }
    return true;
}

bool 
StateFileManager::ReadStateFile(std::string& stateContent)
{
    auto res = readFile(STATE_FILE_PATH  + UM_STATE_FILE, stateContent);
    if (res != 0) {
        UPDATE_LOG_E("ReadStateFile error, code is : %d", res);
        return false;
    }
    return true;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
