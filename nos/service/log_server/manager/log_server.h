#pragma once

#include <thread>

namespace hozon {
namespace netaos {
namespace logserver {

class LogServer {

public:
    LogServer();
    ~LogServer();
    void Init();
    void DeInit();
    void Run();
    void Stop();

private:
    LogServer(const LogServer &);
    LogServer & operator = (const LogServer &);

private:
    bool stop_flag_;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
