/*!
 * @file diag_server_event_mgr.h
 * This file contains the implementation of the diag event manager functions.
 *
 * This file was generated by the tool fastcdrgen.
 */

#include <iostream>
#include <mutex>
#include <memory>

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerEventSub;
class DiagServerEventPub;
class DiagServerEventTest;
class DiagServerEventMgr
{
public:
    DiagServerEventMgr();
    virtual ~DiagServerEventMgr();
    DiagServerEventMgr(DiagServerEventMgr&) = delete;
    DiagServerEventMgr& operator= (const DiagServerEventMgr&) = delete;

    static DiagServerEventMgr* getInstance();
    void Init();
    void DeInit();

private:
    static std::mutex m_mtx;
    static DiagServerEventMgr* m_pInstance;
    std::shared_ptr<DiagServerEventSub> m_spEventSub;
    std::shared_ptr<DiagServerEventPub> m_spEventPub;
    std::shared_ptr<DiagServerEventTest> m_spEventTest;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
