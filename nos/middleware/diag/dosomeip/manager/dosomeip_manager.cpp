
#include "manager/dosomeip_manager.h"
#include "config/dosomeip_config.h"

namespace hozon {
namespace netaos {
namespace diag {

DoSomeIPManager::DoSomeIPManager() {
    skeleton_ = std::make_unique<DoSomeIPSkeleton>();
}

DOSOMEIP_RESULT
DoSomeIPManager::Init(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback, std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback) {
    DS_INFO << "DoSomeIPManager::Init enter!";

    skeleton_->RegistUDSRequestCallback(uds_request_callback);
    skeleton_->RegistSomeIpEstablishCallback(someip_register_callback);

    DS_DEBUG << "load config res is :" << DoSomeIPConfig::Instance()->LoadConfig();
    auto res = skeleton_->Init(DoSomeIPConfig::Instance()->GetInitMaxTimeout());
    if (res) {
        DS_INFO << "DoSomeIPManager::Init on going!";
    } else {
        DS_ERROR << "DoSomeIPManager::Init error!";
        return DOSOMEIP_RESULT::DOSOMEIP_RESULT_INITIAL_FAILED;
    }
    return DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK;
}

void DoSomeIPManager::DeInit() {
    DS_INFO << "DoSomeIPManager::DeInit.";
    skeleton_->Deinit();
    skeleton_->UnRegistUDSRequestCallback();
    skeleton_->UnRegistSomeIpEstablishCallback();
}

DOSOMEIP_RESULT
DoSomeIPManager::DispatchUDSReply(const DoSomeIPRespUdsMessage& udsMsg) {
    auto res = skeleton_->OnDiagProcessComplete(udsMsg);
    if (res) {
        DS_DEBUG << "OnDiagProcessComplete ok!";
    } else {
        DS_ERROR << "OnDiagProcessComplete error!";
        return DOSOMEIP_RESULT::DOSOMEIP_RESULT_ERROR;
    }
    return DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
   /* EOF */