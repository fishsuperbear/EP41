#ifndef DIAG_SERVER_TRANSPORT_METHOD_RECEIVER_H
#define DIAG_SERVER_TRANSPORT_METHOD_RECEIVER_H

#include "cm/include/method.h"
#include "idl/generated/diagPubSubTypes.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

using namespace hozon::netaos::cm;

class DiagServerTransportMethodServer  : public Server<uds_data_method, uds_data_method> {
public:
    DiagServerTransportMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data) : Server(req_data, resp_data) {}
    virtual int32_t Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp);

    virtual ~DiagServerTransportMethodServer();
};

class DiagServerTransportMethodReceiver {
public:
    DiagServerTransportMethodReceiver();
    ~DiagServerTransportMethodReceiver();

    void Init();
    void DeInit();

private:
    std::shared_ptr<DiagServerTransportMethodServer> method_server_;

};

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_METHOD_RECEIVER_H