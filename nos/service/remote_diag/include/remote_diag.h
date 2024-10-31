
#ifndef REMOTE_DIAG_H
#define REMOTE_DIAG_H

namespace hozon {
namespace netaos {
namespace remote_diag {

class RemoteDiag {

public:
    RemoteDiag();
    ~RemoteDiag();
    void Init();
    void DeInit();
    void Run();
    void Stop();

private:
    RemoteDiag(const RemoteDiag &);
    RemoteDiag & operator = (const RemoteDiag &);

private:
    bool stop_flag_;
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // REMOTE_DIAG_H