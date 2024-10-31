#ifndef DIAG_SERVER_TRANSPORT_METHOD_SENDER_H
#define DIAG_SERVER_TRANSPORT_METHOD_SENDER_H

#include <unordered_map>

#include "cm/include/method.h"
#include "idl/generated/diagPubSubTypes.h"
#include "idl/generated/chassis_ota_methodPubSubTypes.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

using namespace hozon::netaos::cm;

class DiagServerTransportMethodSender {
public:
    DiagServerTransportMethodSender();
    ~DiagServerTransportMethodSender();

    void Init();
    void DeInit();

    void DiagMethodSend(const uint8_t sid, const uint8_t subFunc, const std::vector<std::string> service, std::vector<uint8_t>& udsData);
    void ChassisMethodSend();
    bool IsCheckUpdateStatusOk();

private:
    std::unordered_map<std::string, std::shared_ptr<Client<uds_data_method, uds_data_method>>> client_map_;
    std::shared_ptr<Client<ChassisOtaMethod, ChassisOtaMethod>> chassis_info_client_;
    std::shared_ptr<Client<update_status_method, update_status_method>> update_status_client_;

};

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_METHOD_SENDER_H