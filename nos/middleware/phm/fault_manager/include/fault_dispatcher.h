#ifndef FAULT_DISPATCHER_H
#define FAULT_DISPATCHER_H

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include "proxy.h"
#include "skeleton.h"
#include "phmPubSubTypes.h"
#include "phm/include/phm_def.h"
#include "phm/impl/include/phm_client_impl.h"

namespace hozon {
namespace netaos {
namespace phm {

using namespace hozon::netaos::cm;

#define QUEUE_DEPTH 20
#define THREAD_NUMS 1

typedef struct SendFaultPack {
    std::string faultDomain{""};
    uint64_t faultOccurTime{0};
    uint32_t faultId{0};
    uint8_t faultObj{0};
    uint8_t faultStatus{0};
} SendFaultPack_t;


class FaultDispatcher {
public:
    FaultDispatcher();
    ~FaultDispatcher();

    int32_t Init(std::function<void(ReceiveFault_t)> fault_receive_callback);
    void Deinit();

    int32_t InitSend();
    int32_t InitRecv();

    void Send(SendFaultPack_t& sendFault);
    void Recv(void);
    void LocalRecv(SendFaultPack_t& sendFault);

    void NotifyAll();
    void PushFaultQueue(SendFaultPack_t& sendFault);
    void FaultSendThread();

    int32_t InitInhibitTypeRecv();
    void InhibitTypeCb();

private:
    void SystemCheck();
    void SystemCheckCompletedCallback(void* data);

private:
    std::function<void(ReceiveFault_t)> fault_receive_callback_;
    std::shared_ptr<fault_eventPubSubType> pubsubtype_ {nullptr};
    std::shared_ptr<Skeleton> skeleton_ {nullptr};
    std::shared_ptr<Proxy> proxy_ {nullptr};
    std::shared_ptr<fault_event> event_data_ {nullptr};

    std::shared_ptr<faultInhibitEventPubSubType> fault_inhibit_pubsubtype_ {nullptr};
    std::shared_ptr<Proxy> inhibit_type_proxy_ {nullptr};
    std::shared_ptr<faultInhibitEvent> fault_inhibit_data_ {nullptr};

    // system check
    int system_check_timer_fd_;
    bool system_check_completed_;

    std::function<void(SendFaultPack_t&)> local_receiver_callback_;
    std::unordered_map<uint16_t, bool> local_receive_flag_;
    std::queue<SendFaultPack_t> faultq_;
    std::mutex mutexs_[THREAD_NUMS];
    std::condition_variable cvs_[THREAD_NUMS];
    std::thread dispatcher_threads_[THREAD_NUMS];
    bool stopFlag_;
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_DISPATCHER_H
