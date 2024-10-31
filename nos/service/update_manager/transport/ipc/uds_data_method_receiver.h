#ifndef UDS_DATA_METHOD_RECEIVER_H
#define UDS_DATA_METHOD_RECEIVER_H

#include "update_manager/common/data_def.h"
#include "diag/ipc/api/ipc_server.h"
#include "diag/ipc/proto/diag.pb.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::diag;

class UdsDataMethodServer : public IPCServer
{
public:
    virtual ~UdsDataMethodServer() {};
    virtual int32_t Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp);
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