#include <thread>
#include "diag/diag_server/include/diag_server.h"
#include "diag/diag_server/include/common/diag_server_life_mgr.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/common/function_statistics.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerLifeMgr life_mgr;

DiagServer::DiagServer()
: stop_flag_(false)
{
}

DiagServer::~DiagServer()
{
}

void
DiagServer::Init()
{
    DG_INFO << "DiagServer::Init";
    FunctionStatistics func("DiagServer::Init finish, ");
    life_mgr.Init();
}

void
DiagServer::DeInit()
{
    DG_INFO << "DiagServer::DeInit";
    FunctionStatistics func("DiagServer::DeInit finish, ");
    life_mgr.DeInit();
}

void
DiagServer::Run()
{
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    DeInit();
}

void
DiagServer::Stop()
{
    stop_flag_ = true;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
