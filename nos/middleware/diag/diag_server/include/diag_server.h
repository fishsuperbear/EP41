
#ifndef DIAG_SERVER_H
#define DIAG_SERVER_H

namespace hozon {
namespace netaos {
namespace diag {

class DiagServer {

public:
    DiagServer();
    ~DiagServer();
    void Init();
    void DeInit();
    void Run();
    void Stop();

private:
    DiagServer(const DiagServer &);
    DiagServer & operator = (const DiagServer &);

private:
    bool stop_flag_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_H