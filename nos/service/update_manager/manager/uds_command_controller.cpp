#include "update_manager/manager/uds_command_controller.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

UdsCommandController* UdsCommandController::m_pInstance = nullptr;
std::mutex UdsCommandController::m_mtx;

UdsCommandController::UdsCommandController()
: nextExpectedIndex{0}
{
    expectedCommands = {"2E F1 98", "2E F1 99"};
}

UdsCommandController::~UdsCommandController()
{
}

UdsCommandController*
UdsCommandController::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new UdsCommandController();
        }
    }

    return m_pInstance;
}

void
UdsCommandController::Init()
{
    UM_INFO << "UdsCommandController::Init.";
    UM_INFO << "UdsCommandController::Init Done.";
}

void
UdsCommandController::Deinit()
{
    UM_INFO << "UdsCommandController::Deinit.";
    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "UdsCommandController::Deinit Done.";
}

bool
UdsCommandController::ProcessCommand(const std::string& command)
{
    UPDATE_LOG_D("command : %s, expectedCommands[nextExpectedIndex]: %s ", command.c_str(), expectedCommands[nextExpectedIndex].c_str());
    if (command == expectedCommands[nextExpectedIndex]) {
        nextExpectedIndex = (nextExpectedIndex + 1) % expectedCommands.size();
        return true;
    } else {
        if (command == "2E F1 98" || command == "2E F1 99")
        {
            nextExpectedIndex = (nextExpectedIndex + 1) % expectedCommands.size();
            return true;
        }
        nextExpectedIndex = 0;
        return false;
    }
}

bool 
UdsCommandController::ResetProcess()
{
    nextExpectedIndex = 0;
    return true;
}

bool 
UdsCommandController::IsSocVersionSame()
{
    return isVersionSame;
}

bool 
UdsCommandController::SetVersionSameFlag(bool flag)
{
    isVersionSame = flag;
    return true;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
