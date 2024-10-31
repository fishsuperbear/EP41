#pragma once

#include <list>
#include "log_server/common/log_server_def.h"
#include "log/src/spdlog/include/spdlog/common.h"

#include "zmq_ipc/manager/zmq_ipc_server.h"
#include "zmq_ipc/proto/log_server.pb.h"

namespace hozon {
namespace netaos {
namespace logserver {

class CompressLogImpl final : public hozon::netaos::zmqipc::ZmqIpcServer
{

public:
    CompressLogImpl();
    virtual ~CompressLogImpl(){};
    int32_t Init();
    int32_t DeInit();
    virtual int32_t Process(const std::string& request, std::string& reply);

private:
    void CompressFile(const std::string& request);
    bool rename_file_(const spdlog::filename_t &src_filename, const spdlog::filename_t &target_filename);
    void StartProcessThread();
private:
    bool stopFlag_;
    std::list<std::string>  request_queue_;
    std::mutex              request_queue_mutex_;
    std::vector<std::thread> process_threads_pool_;
    std::condition_variable process_condition_;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
