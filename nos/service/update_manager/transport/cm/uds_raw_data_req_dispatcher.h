#ifndef UDS_RAW_DATA_REQ_DISPATCHER_H
#define UDS_RAW_DATA_REQ_DISPATCHER_H

#include "cm/include/skeleton.h"
#include "idl/generated/diagPubSubTypes.h"


namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::cm;

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
    void Send(UdsRawDataReqEvent& sendUdsRawDataReq);
    void Deinit();

private:
    std::shared_ptr<Skeleton> skeleton_;

};


}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UDS_RAW_DATA_REQ_DISPATCHER_H
