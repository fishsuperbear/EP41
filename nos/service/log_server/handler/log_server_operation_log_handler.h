#pragma once

#include <mutex>
#include <memory>
#include "log_server/impl/operation_log_impl.h"

namespace hozon {
namespace netaos {
namespace logserver {

class logServerOperationLogHandler {

public:
    static logServerOperationLogHandler* getInstance();

    void Init();
    void DeInit();

private:
    logServerOperationLogHandler();
    logServerOperationLogHandler(const logServerOperationLogHandler &);
    logServerOperationLogHandler & operator = (const logServerOperationLogHandler &);

private:
    static std::mutex mtx_;
    static logServerOperationLogHandler* instance_;
    
    std::unique_ptr<OperationLogImpl> impl_;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
