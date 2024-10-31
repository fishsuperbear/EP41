
#ifndef DIAG_SERVER_SERVICE_DATA_HANDLER_H
#define DIAG_SERVER_SERVICE_DATA_HANDLER_H


#include "diag/diag_server/include/common/diag_server_def.h"
#include <mutex>

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsDataHandler {

public:
    static DiagServerUdsDataHandler* getInstance();

    void Init();
    void DeInit();

    void RecvUdsMessage(DiagServerUdsMessage& udsMessage);
    void ReplyUdsMessage(const DiagServerUdsMessage& udsMessage);

    void NotifyMessageFailure(const DiagServerUdsMessage& udsMessage);
    void TransmitConfirmation(const DiagServerUdsMessage& udsMessage, const bool confirmResult);

private:
    DiagServerUdsDataHandler();
    DiagServerUdsDataHandler(const DiagServerUdsDataHandler &);
    DiagServerUdsDataHandler & operator = (const DiagServerUdsDataHandler &);

private:
    static DiagServerUdsDataHandler* instance_;
    static std::mutex mtx_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_SERVICE_DATA_HANDLER_H
