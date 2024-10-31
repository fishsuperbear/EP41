#include "log_server/common/log_server_life_mgr.h"
#include "log_server/log/log_server_logger.h"
#include "log_server/handler/log_server_operation_log_handler.h"
#include "log_server/handler/log_server_compress_handler.h"
#include "log_server/handler/log_server_mcu_handler.h"
#include "log_server/handler/log_server_fault_handler.h"
namespace hozon {
namespace netaos {
namespace logserver {

LogServerLifeMgr::LogServerLifeMgr()
{
}

LogServerLifeMgr::~LogServerLifeMgr()
{
}

void
LogServerLifeMgr::Init()
{
    LOG_SERVER_INFO << "LogServerLifeMgr::Init.";
    logServerFaultHandler::getInstance()->Init();
    logServerOperationLogHandler::getInstance()->Init();
    logServerCompressHandler::getInstance()->Init();
    logServerMcuHandler::getInstance()->Init();
    LOG_SERVER_INFO << "LogServerLifeMgr::Init Done.";
}

void
LogServerLifeMgr::DeInit()
{
    LOG_SERVER_INFO << "LogServerLifeMgr::DeInit.";
    logServerMcuHandler::getInstance()->DeInit();
    logServerCompressHandler::getInstance()->DeInit();
    logServerOperationLogHandler::getInstance()->DeInit();
    logServerFaultHandler::getInstance()->DeInit();
    LOG_SERVER_INFO << "LogServerLifeMgr::DeInit Done.";
}

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
