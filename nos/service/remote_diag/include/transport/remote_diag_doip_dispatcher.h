#ifndef REMOTE_DIAG_DOIP_DISPATCHER_H
#define REMOTE_DIAG_DOIP_DISPATCHER_H

#include "diag/doip/include/api/doip_transport.h"
#include "remote_diag/include/common/remote_diag_def.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

using namespace hozon::netaos::diag;

enum SpeedGetStatus {
    Default = 0x01,
    Progess = 0x02,
    Success = 0x03,
    Failed = 0x04
};

class RemoteDiagDoipDispatcher {
public:
    RemoteDiagDoipDispatcher();
    ~RemoteDiagDoipDispatcher();

    void Init();
    void DeInit();

    void DoipRequestByEquip(const RemoteDiagReqUdsMessage& udsMessage, const bool speedFlag = false);
    bool GetVehicleSpeed();

private:
    void DoipConfirmCallback(doip_confirm_t* doipConfirm);
    void DoipIndicationCallback(doip_indication_t* doipIndication);
    void DoipRouteCallback(doip_route_t* doipRoute) {}

private:
    DoIPTransport* service_doip_;

    SpeedGetStatus speed_get_status_;
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // REMOTE_DIAG_DOIP_DISPATCHER_H