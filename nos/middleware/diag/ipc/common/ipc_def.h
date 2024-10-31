#pragma once

#include <memory>
#include <map>
#include <vector>
#include <string>
#include <thread>
#include <functional>
#include <future>

#include "libipc/ipc.h"

#include "diag/ipc/log/ipc_logger.h"
#include "diag/ipc/common/ipc_funtions.h"

namespace hozon {
namespace netaos {
namespace diag {


// 定义一个多对多的通讯管道
using msg_line = ipc::chan<ipc::relat::multi, ipc::relat::multi, ipc::trans::broadcast>;

const std::string prefix = "__IPC_SHM___";
const std::string default_path = "/dev/shm/";

}  // namespace diag
}  // namespace netaos
}  // namespace hozon