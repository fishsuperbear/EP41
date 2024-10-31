#include "log_server/manager/log_server.h"
#include "log_server/log/log_server_logger.h"
#include "log_server/common/log_server_life_mgr.h"
#include "log_server/common/function_statistics.h"

namespace hozon {
namespace netaos {
namespace logserver {

LogServerLifeMgr life_mgr;

LogServer::LogServer()
: stop_flag_(false)
{
}

LogServer::~LogServer()
{
}

void
LogServer::Init()
{
    LOG_SERVER_INFO << "LogServer::Init.";
    FunctionStatistics func("LogServer::Init Done, ");
    life_mgr.Init();
}

void
LogServer::DeInit()
{
    LOG_SERVER_INFO << "LogServer::DeInit.";
    FunctionStatistics func("LogServer::DeInit Done, ");
    life_mgr.DeInit();
}

void
LogServer::Run()
{
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    DeInit();
}

void
LogServer::Stop()
{
    stop_flag_ = true;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon