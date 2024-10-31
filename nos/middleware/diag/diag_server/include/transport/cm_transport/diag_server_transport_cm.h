
#ifndef DIAG_SERVER_TRANSPORT_CM_H
#define DIAG_SERVER_TRANSPORT_CM_H

#include <mutex>
#include <iostream>

#include "diag/common/include/thread_pool.h"
#include "diag_server_transport_event_sender.h"
#include "diag_server_transport_event_receiver.h"
#include "diag_server_transport_method_sender.h"
#include "diag_server_transport_method_receiver.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

class DiagServerTransPortCM {

public:
    static DiagServerTransPortCM* getInstance();

    void Init();
    void DeInit();

    // diag event
    void DiagEventSend(const DiagServerRespUdsMessage& udsMessage, const bool remote = false);
    void DiagEventCallback(const DiagServerReqUdsMessage& udsMessage);
    void DiagSessionEventSend(const DiagServerSessionCode& session);

    // method
    void DiagMethodSend(const uint8_t sid, const uint8_t subFunc, const std::vector<std::string> service, std::vector<uint8_t>& udsData);
    void ChassisMethodSend();
    bool IsCheckUpdateStatusOk();

private:
    DiagServerTransPortCM();
    DiagServerTransPortCM(const DiagServerTransPortCM &);
    DiagServerTransPortCM & operator = (const DiagServerTransPortCM &);

private:
    static DiagServerTransPortCM* instance_;
    static std::mutex mtx_;

    std::unique_ptr<ThreadPool> threadpool_;

    DiagServerTransportEventSender* event_sender_;
    DiagServerTransportEventReceiver* event_receiver_;
    DiagServerTransportMethodSender* method_sender_;
    DiagServerTransportMethodReceiver* method_receiver_;
};

class UdsDataTask : public BaseTask {

public:
    UdsDataTask() = default;
    UdsDataTask(std::string taskName) : BaseTask(taskName) {};
    ~UdsDataTask();
    int Run();
};

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_CM_H