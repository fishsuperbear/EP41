#pragma once

#include "update_manager/common/data_def.h"
#include "diag/ipc/api/ipc_server.h"
#include "diag/ipc/proto/diag.pb.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::diag;
class UpdateStatusMethodServer : public IPCServer 
{
public:
    virtual ~UpdateStatusMethodServer() {};
    virtual int32_t Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp);
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