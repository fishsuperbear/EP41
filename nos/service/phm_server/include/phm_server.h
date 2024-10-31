
#ifndef PHM_SERVER_H
#define PHM_SERVER_H

namespace hozon {
namespace netaos {
namespace phm_server {

class PhmServer {

public:
    PhmServer();
    ~PhmServer();
    void Init();
    void DeInit();
    void Run();
    void Stop();

private:
    PhmServer(const PhmServer &);
    PhmServer & operator = (const PhmServer &);

private:
    bool stop_flag_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_SERVER_H