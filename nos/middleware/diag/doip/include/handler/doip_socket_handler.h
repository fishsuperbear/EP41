/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip socket handler
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_SOCKET_HANDLER_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_SOCKET_HANDLER_H_

#include <list>
#include <memory>
#include <functional>

#include "diag/doip/include/handler/doip_message_handler.h"
#include "diag/doip/include/base/doip_connection.h"
#include "diag/doip/include/base/doip_thread.h"
#include "diag/doip/include/base/doip_netlink.h"
#include "diag/doip/include/base/doip_event_loop.h"
#include "diag/doip/include/socket/doip_socket.h"
#include "diag/doip/include/config/doip_config.h"


namespace hozon {
namespace netaos {
namespace diag {

class DoIPSocketHandler;

typedef struct doip_client {
    DoIPSocketHandler *handler;
    DoipConnection *connection;
    doip_event_source_t *source;
    uint8_t client_type;
} doip_client_t;



class DoIPSocketHandler {
 public:
    DoIPSocketHandler();
    ~DoIPSocketHandler();
    doip_result_t Init(std::function<void(doip_indication_t*)> indication_callback,
                       std::function<void(doip_confirm_t*)>    confirm_callback,
                       std::function<void(doip_route_t*)>      route_callback,
                       std::string doip_config);
    void Deinit();
    doip_result_t Start();
    void Stop();

    void DoipRegistReleaseCallback(std::function<void(doip_netlink_status_t, uint16_t)> release_callback);

    doip_result_t DoipRequestByNode(const doip_request_t* request);
    doip_result_t DoipRequestByEquip(const doip_request_t* request);
    doip_result_t DoipReleaseByEquip(const doip_request_t* request);

    ssize_t SocketWrite(DoipConnection *&connection);
    ssize_t SocketWriteRaw(int32_t fd, const char* buffer, ssize_t buffer_size);
    ssize_t SocketRead(DoipConnection *&connection);
    ssize_t SocketSendto(DoipConnection *connection, const struct sockaddr *addr, int32_t addr_len);
    ssize_t SocketSendto(DoipConnection *connection, char *ip, uint16_t port);
    ssize_t SocketRecvfrom(DoipConnection *&connection, struct sockaddr *addr, socklen_t *size);

    int32_t Ipv4TcpCreate(char *ip, uint16_t port);
    int32_t DoipClientRemove(int32_t fd);
    void DoipClientRemoveAll();

    DoipConnection* GetUDPSConnection(int32_t fd);
    DoipConnection* GetUDPCConnection();
    static uint8_t GetNCTS();

    std::vector<DoipIpv4Socket> ipv4_tcp_socket_;
    std::vector<DoipIpv4Socket> ipv4_udps_socket_;
    DoipIpv4Socket ipv4_udpc_socket_;

 private:
    void ThreadMain(void *arg);
    void ThreadNetlinkMonitor(void *arg);
    doip_client_t* DoipClientCreate(int32_t fd, int8_t type);
    void ClientDestroy(doip_client_t* client);
    int32_t NetlinkMonitorCreate();
    void IpAddrAssignment(doip_net_source_t* net_source);
    int32_t AddIpv4TcpSocket(doip_net_source_t* net_source);
    int32_t AddIpv4DoipserverUdpSocket(doip_net_source_t* net_source);
    int32_t AddIpv4DoipclientUdpSocket(const doip_net_source_t* net_source);
    int32_t ConnectSocketData(int32_t fd, uint32_t mask, void *data);
    int32_t UdpsSocketData(int32_t fd, uint32_t mask, void *data);
    int32_t UdpcSocketData(int32_t fd, uint32_t mask, void *data);
    int32_t TcpSocketData(int32_t fd, uint32_t mask, void *data);

    std::shared_ptr<DoIPMessageHandler> message_handler_;
    std::shared_ptr<DoipEventLoop> event_loop_;
    std::unique_ptr<DoipNetlink> net_link_;
    std::mutex net_client_list_mutex_;
    static std::list<doip_client_t*> net_client_list_;
    static std::list<doip_client_t*> in_client_list_;
    std::vector<doip_client_t *> udps_virtual_client_;
    doip_client_t* udpc_virtual_client_{nullptr};
    doip_thread_t* thread_main_{nullptr};
    doip_thread_t* thread_netlink_monitor_{nullptr};
    bool init_flag_{false};
    std::function<void(doip_netlink_status_t, uint16_t)> release_callback_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_SOCKET_HANDLER_H_
