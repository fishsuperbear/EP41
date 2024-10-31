#pragma once

#include "cm/include/method.h"
#include "idl/generated/diagPubSubTypes.h"
#include "update_manager/common/data_def.h"


namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::cm;

class UpdateStatusMethodServer : public Server<update_status_method, update_status_method> {
public:
    UpdateStatusMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data) : Server(req_data, resp_data) {}
    virtual ~UpdateStatusMethodServer() {};
    virtual int32_t Process(const std::shared_ptr<update_status_method> req, std::shared_ptr<update_status_method> resp);
};

class UpdateStatusMethodReceiver {
public:
    UpdateStatusMethodReceiver();
    ~UpdateStatusMethodReceiver();

    void Init();
    void DeInit();

private:
    std::shared_ptr<UpdateStatusMethodServer> method_server_;

};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon