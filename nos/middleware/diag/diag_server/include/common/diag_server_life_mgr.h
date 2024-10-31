
#ifndef DIAG_SERVER_LIFE_MGR_H
#define DIAG_SERVER_LIFE_MGR_H

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerLifeMgr {

public:
    DiagServerLifeMgr();
    ~DiagServerLifeMgr();
    void Init();
    void DeInit();

private:
    DiagServerLifeMgr(const DiagServerLifeMgr &);
    DiagServerLifeMgr & operator = (const DiagServerLifeMgr &);
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_LIFE_MGR_H