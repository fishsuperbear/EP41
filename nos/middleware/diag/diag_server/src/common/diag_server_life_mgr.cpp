#include "diag/diag_server/include/transport/diag_server_transport.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/common/diag_server_life_mgr.h"
#include "diag/diag_server/include/uds/diag_server_uds_mgr.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/info/diag_server_dynamic_info.h"
#include "diag/diag_server/include/info/diag_server_stored_info.h"
#include "diag/diag_server/include/info/diag_server_chassis_info.h"
#include "diag/diag_server/include/security/diag_server_security_mgr.h"
#include "diag/diag_server/include/transport/diag_server_transport_service.h"
#include "diag/diag_server/include/datatransfer/diag_server_data_transfer.h"
#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/event_manager/diag_server_event_mgr.h"
#include "diag/diag_server/include/publish/diag_server_uds_pub.h"


namespace hozon {
namespace netaos {
namespace diag {

DiagServerLifeMgr::DiagServerLifeMgr()
{
}

DiagServerLifeMgr::~DiagServerLifeMgr()
{
}

void
DiagServerLifeMgr::Init()
{
    DG_DEBUG << "DiagServerLifeMgr::Init";
    DiagServerConfig::getInstance()->Init();
#ifdef BUILD_FOR_ORIN
    DiagServerUdsPub::getInstance()->Init();
#endif
    DiagServerChassisInfo::getInstance()->Init();
    DiagServerStoredInfo::getInstance()->Init();
    DiagServerDynamicInfo::getInstance()->Init();
    DiagServerDataTransfer::getInstance()->Init();
    DiagServerSecurityMgr::getInstance()->Init();
    DiagServerSessionHandler::getInstance()->Init();
    DiagServerTransPortService::getInstance()->Init();
    DiagServerTransport::getInstance()->Init();
    DiagServerUdsDataHandler::getInstance()->Init();
    DiagServerUdsMgr::getInstance()->Init();
    DiagServerEventMgr::getInstance()->Init();
}

void
DiagServerLifeMgr::DeInit()
{
    DG_DEBUG << "DiagServerLifeMgr::DeInit";
    DiagServerEventMgr::getInstance()->DeInit();
    DiagServerUdsMgr::getInstance()->DeInit();
    DiagServerUdsDataHandler::getInstance()->DeInit();
    DiagServerTransport::getInstance()->DeInit();
    DiagServerTransPortService::getInstance()->DeInit();
    DiagServerSessionHandler::getInstance()->DeInit();
    DiagServerSecurityMgr::getInstance()->DeInit();
    DiagServerDataTransfer::getInstance()->DeInit();
    DiagServerDynamicInfo::getInstance()->DeInit();
    DiagServerStoredInfo::getInstance()->DeInit();
    DiagServerChassisInfo::getInstance()->DeInit();
#ifdef BUILD_FOR_ORIN
    DiagServerUdsPub::getInstance()->DeInit();
#endif
    DiagServerConfig::getInstance()->DeInit();
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
