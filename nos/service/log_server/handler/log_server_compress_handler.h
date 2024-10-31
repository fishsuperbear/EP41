#pragma once

#include <mutex>
#include <memory>
#include "log_server/impl/compress_log_impl.h"

namespace hozon {
namespace netaos {
namespace logserver {

class logServerCompressHandler {

public:
    static logServerCompressHandler* getInstance();

    void Init();
    void DeInit();

private:
    logServerCompressHandler();
    logServerCompressHandler(const logServerCompressHandler &);
    logServerCompressHandler & operator = (const logServerCompressHandler &);

private:
    static std::mutex mtx_;
    static logServerCompressHandler* instance_;
    
    std::unique_ptr<CompressLogImpl> impl_;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
