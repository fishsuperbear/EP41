#ifndef UDS_RAW_DATA_RESP_RECEIVER_H
#define UDS_RAW_DATA_RESP_RECEIVER_H

#include "cm/include/proxy.h"
#include "idl/generated/diagPubSubTypes.h"
#include "update_manager/common/data_def.h"


namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::cm;

class UdsRawDataRespReceiver {
public:
    UdsRawDataRespReceiver();
    ~UdsRawDataRespReceiver();

    void Init();
    void DeInit();

    void RegistUdsRawDataReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback);
    void RegistReadVersionReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> read_version_callback);

    void EventCallback();

private:
    std::shared_ptr<Proxy> proxy_;
    std::shared_ptr<uds_raw_data_resp_event> data_;
    std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback_;
    std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> read_version_callback_;

};


}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UDS_RAW_DATA_RESP_RECEIVER_H