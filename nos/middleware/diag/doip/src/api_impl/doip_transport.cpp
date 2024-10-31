/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip transport
 */
#include "diag/doip/include/api/doip_transport.h"
#include "diag/doip/include/base/doip_logger.h"


namespace hozon {
namespace netaos {
namespace diag {


DoIPTransport::DoIPTransport() {
    DoIPLogger::GetInstance().CreateLogger("doip");
    socket_handler_ = std::make_shared<DoIPSocketHandler>();
}

DoIPTransport::~DoIPTransport() {
}

doip_result_t
DoIPTransport::DoipInit(std::function<void(doip_indication_t*)> indication_callback,
                        std::function<void(doip_confirm_t*)>    confirm_callback,
                        std::function<void(doip_route_t*)>      route_callback,
                        std::string doip_config) {
    DOIP_INFO << "DoIPTransport::DoipInit enter!";

    doip_result_t res = socket_handler_->Init(indication_callback, confirm_callback, route_callback, doip_config);
    if (res == DOIP_RESULT_OK) {
        res = socket_handler_->Start();
        if (res == DOIP_RESULT_OK) {
            DOIP_INFO << "DoIPTransport::DoipInit finish!";
        } else {
            DOIP_ERROR << "DoIPTransport::DoipInit error!";
        }
    } else {
        DOIP_ERROR << "DoIPTransport::DoipInit error!";
    }

    return res;
}

void
DoIPTransport::DoipDeinit() {
    DOIP_INFO << "DoIPTransport::DoipDeinit enter!";
    socket_handler_->Stop();
    socket_handler_->Deinit();
    DOIP_INFO << "DoIPTransport::DoipDeinit finish!";
}

void
DoIPTransport::DoipRegistReleaseCallback(std::function<void(doip_netlink_status_t, uint16_t)> release_callback) {
    socket_handler_->DoipRegistReleaseCallback(release_callback);
}

doip_result_t
DoIPTransport::DoipRequestByNode(const doip_request_t* request) {
    return socket_handler_->DoipRequestByNode(request);
}

doip_result_t
DoIPTransport::DoipRequestByEquip(const doip_request_t* request) {
    return socket_handler_->DoipRequestByEquip(request);
}

doip_result_t
DoIPTransport::DoipReleaseByEquip(const doip_request_t* request)
{
    return socket_handler_->DoipReleaseByEquip(request);
}



}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
