/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip message handler
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_HANDLER_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_HANDLER_H_

#include <string>
#include <memory>
#include <mutex>
#include <list>
#include <functional>

#include "diag/common/include/thread_pool.h"
#include "diag/common/include/timer_manager.h"
#include "diag/doip/include/base/doip_connection.h"
#include "diag/doip/include/data_type/doip_def_internal.h"
#include "diag/doip/include/config/doip_config.h"


namespace hozon {
namespace netaos {
namespace diag {


typedef struct doip_connect_data {
    int32_t fd;
    char ip[INET6_ADDRSTRLEN];
    uint16_t port;
    uint8_t comm_type;
} doip_connect_data_t;


class DoIPSocketHandler;

class DoIPMessageHandler {
 public:
    DoIPMessageHandler();
    ~DoIPMessageHandler();
    void Init(DoIPSocketHandler* socket_handler,
              std::function<void(doip_indication_t*)> indication_callback,
              std::function<void(doip_confirm_t*)>    confirm_callback,
              std::function<void(doip_route_t*)>      route_callback);
    void DeInit();
    void DoipRegistReleaseCallback(std::function<void(doip_netlink_status_t, uint16_t)> release_callback);

    void AddConnectionTable(doip_connect_data_t* connect_data);
    void RemoveConnectionTable(int32_t fd, uint8_t socket_type);
    void DoipMessageUnpack(doip_link_data_t* link_data);

    void DoipVehicleAnnounceStart(const std::string& if_name);
    void DoipVehicleAnnounceStop(const std::string& if_name);

    doip_result_t DoipRequestByNode(const doip_request_t* request);
    doip_result_t DoipRequestByEquip(const doip_request_t* request);
    doip_result_t DoipReleaseByEquip(const doip_request_t* request);

    std::function<void(doip_netlink_status_t, uint16_t)> release_callback_;

 private:
    void DoipMessageUnpackExec(doip_link_data_t* link_data);
    uint32_t DoipMessageUnpackExecUdpserver(doip_link_data_t* link_data, uint32_t pos);
    uint32_t DoipMessageUnpackExecUdpclient(doip_link_data_t* link_data, uint32_t pos);
    uint32_t DoipMessageUnpackExecTcpserver(doip_link_data_t* link_data, uint32_t pos);
    uint32_t DoipMessageUnpackExecTcpclient(doip_link_data_t* link_data, uint32_t pos);

    int PerformRoutingActivation(char *ip, uint16_t port, uint16_t logical_source_address);
    uint32_t MessageTaskAdd(doip_link_data_t* link_data, std::string type, uint32_t dependentCID = 0);

    void AnnounceWaitTimerCallback(void* data);
    void AnnounceIntervalTimerCallback(void* data);
    void TCPInitialInactivityTimerCallback(void* data);
    void TCPGeneralInactivityTimerCallback(void* data);

    std::unique_ptr<ThreadPool> threadpool_;
    std::shared_ptr<TimerManager> timerManager_;
    DoIPSocketHandler* socket_handler_;
    std::mutex mtx_;
    std::mutex equip_mtx_;
    std::list<doip_node_tcp_table_t*> node_tcp_tables_;
    std::list<doip_node_udp_table_t*> node_udp_tables_;
    std::list<doip_equip_tcp_table_t*> equip_tcp_tables_;
    std::list<doip_equip_udp_table_t*> equip_udp_tables_;

    std::function<void(doip_indication_t*)> indication_callback_;
    std::function<void(doip_confirm_t*)>    confirm_callback_;
    std::function<void(doip_route_t*)>      route_callback_;

    friend class DoIPMessageUnPack;
    friend class DoIPMessagePack;

    class MessageTask : public BaseTask {
     public:
        MessageTask() = default;
        MessageTask(DoIPMessageHandler* handler, std::string taskName) : BaseTask(taskName), message_handler_(handler) {}
        ~MessageTask();
        int Run();
     private:
        DoIPMessageHandler* message_handler_;
    };
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_HANDLER_H_
