#ifndef UDS_RAW_DATA_RESP_RECEIVER_H
#define UDS_RAW_DATA_RESP_RECEIVER_H

#include "update_manager/common/data_def.h"
#include "diag/ipc/api/ipc_server.h"
#include "diag/ipc/proto/diag.pb.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::diag;

class UdsRawDataRespReceiver : public IPCServer 
{
public:
    UdsRawDataRespReceiver();
    virtual ~UdsRawDataRespReceiver();

    void Init();
    void DeInit();

    virtual int32_t Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp);

    void RegistUdsRawDataReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback);

private:
    std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback_;

};


}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UDS_RAW_DATA_RESP_RECEIVER_H