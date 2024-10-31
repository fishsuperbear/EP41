#include <stdlib.h>
#include <unordered_map>
#include "diag/diag_agent/include/impl/diag_agent_handler_impl.h"
#include "diag/diag_agent/include/handler/diag_agent_handler.h"
#include "diag/diag_agent/include/common/diag_agent_logger.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent{

DiagAgentHandler::DiagAgentHandler()
{
    handler_impl_ = std::make_unique<DiagAgentHandlerImpl>();
}

DiagAgentHandler::~DiagAgentHandler()
{
}

DiagAgentInitResultCode
DiagAgentHandler::Init(const std::string& configPath,
                       std::shared_ptr<DiagAgentDataIdentifier> dataIdentifier,
                       std::shared_ptr<DiagAgentRoutineControl> routineControl)
{
    DiagAgentLogger::GetInstance().CreateLogger("DGA");
    if (nullptr == handler_impl_) {
        return DiagAgentInitResultCode::kHandlerImplNull;
    }

    return handler_impl_->Init(configPath, dataIdentifier, routineControl);
}

void
DiagAgentHandler::DeInit()
{
    if (nullptr == handler_impl_) {
        return;
    }

    handler_impl_->DeInit();
}

}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon