#pragma once

namespace hozon {
namespace netaos {
namespace logserver {

class LogServerLifeMgr {

public:
    LogServerLifeMgr();
    ~LogServerLifeMgr();
    void Init();
    void DeInit();

private:
    LogServerLifeMgr(const LogServerLifeMgr &);
    LogServerLifeMgr & operator = (const LogServerLifeMgr &);
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
