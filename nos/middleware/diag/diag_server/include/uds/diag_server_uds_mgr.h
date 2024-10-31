
#ifndef DIAG_SERVER_UDS_MGR_H
#define DIAG_SERVER_UDS_MGR_H

#include <unordered_map>
#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsMgr {

public:
    static DiagServerUdsMgr* getInstance();

    void Init();
    void DeInit();

    void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);
    void sendNegativeResponse(DiagServerServiceRequestOpc eOpc, DiagServerUdsMessage& udsMessage);
    void sendPositiveResponse(DiagServerServiceRequestOpc eOpc, DiagServerUdsMessage& udsMessage);
    DiagServerUdsBase* getSidService(DiagServerServiceRequestOpc eOpc);

private:
    DiagServerUdsMgr();
    DiagServerUdsMgr(const DiagServerUdsMgr &);
    DiagServerUdsMgr & operator = (const DiagServerUdsMgr &);

private:
    static DiagServerUdsMgr* instance_;
    static std::mutex mtx_;

    // map<sid, service>
    std::unordered_map<uint8_t, DiagServerUdsBase*> sid_base_map_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_MGR_H
