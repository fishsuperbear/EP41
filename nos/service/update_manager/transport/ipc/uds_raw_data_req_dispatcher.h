#ifndef UDS_RAW_DATA_REQ_DISPATCHER_H
#define UDS_RAW_DATA_REQ_DISPATCHER_H

#include "diag/ipc/api/ipc_client.h"
#include "diag/ipc/proto/diag.pb.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::diag;


struct UdsRawDataReqEvent
{
    uint16_t sa;
    uint16_t ta;
    uint8_t bus_type;
    std::vector<uint8_t> data_vec;
};

class UdsRawDataReqDispatcher {
public:
    UdsRawDataReqDispatcher();
    ~UdsRawDataReqDispatcher();

    void Init();
    void Deinit();

    void Send(UdsRawDataReqEvent& sendUdsRawDataReq);

private:
    std::shared_ptr<IPCClient> client_;

};


}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UDS_RAW_DATA_REQ_DISPATCHER_H
