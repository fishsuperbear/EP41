#include <thread>

#include "remote_diag/include/remote_diag.h"
#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/handler/remote_diag_handler.h"
#include "remote_diag/include/extension/remote_diag_file_transfer.h"
#include "remote_diag/include/extension/remote_diag_dynamic_plugin.h"
#include "remote_diag/include/extension/remote_diag_switch_control.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

RemoteDiag::RemoteDiag()
: stop_flag_(false)
{
}

RemoteDiag::~RemoteDiag()
{
}

void
RemoteDiag::Init()
{
    DGR_INFO << "RemoteDiag::Init";
    RemoteDiagConfig::getInstance()->Init();
    RemoteDiagFileTransfer::getInstance()->Init();
    RemoteDiagDynamicPlugin::getInstance()->Init();
    RemoteDiagSwitchControl::getInstance()->Init();
    RemoteDiagHandler::getInstance()->Init();
}

void
RemoteDiag::DeInit()
{
    DGR_INFO << "RemoteDiag::DeInit";
    RemoteDiagHandler::getInstance()->DeInit();
    RemoteDiagSwitchControl::getInstance()->DeInit();
    RemoteDiagDynamicPlugin::getInstance()->DeInit();
    RemoteDiagFileTransfer::getInstance()->DeInit();
    RemoteDiagConfig::getInstance()->DeInit();
}

void
RemoteDiag::Run()
{
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    DeInit();
}

void
RemoteDiag::Stop()
{
    stop_flag_ = true;
}

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
