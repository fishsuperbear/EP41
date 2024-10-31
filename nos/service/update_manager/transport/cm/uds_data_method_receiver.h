#ifndef UDS_DATA_METHOD_RECEIVER_H
#define UDS_DATA_METHOD_RECEIVER_H

#include "cm/include/method.h"
#include "idl/generated/diagPubSubTypes.h"
#include "update_manager/common/data_def.h"


namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::cm;

class UdsDataMethodServer : public Server<uds_data_method, uds_data_method> {
public:
    UdsDataMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data) : Server(req_data, resp_data) {}
    virtual ~UdsDataMethodServer() {};
    virtual int32_t Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp);
};

class UdsDataMethodReceiver {
public:
    UdsDataMethodReceiver();
    ~UdsDataMethodReceiver();

    void Init();
    void DeInit();

private:
    std::shared_ptr<UdsDataMethodServer> method_server_;

};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // UDS_DATA_METHOD_RECEIVER_H