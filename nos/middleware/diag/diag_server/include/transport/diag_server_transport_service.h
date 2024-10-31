
#ifndef DIAG_SERVER_TRANSPORT_SERVICE_H
#define DIAG_SERVER_TRANSPORT_SERVICE_H

#include <mutex>
#include <iostream>
#include <functional>

#include "diag/docan/include/docan_service.h"
#include "diag/docan/include/docan_listener.h"
#include "diag/docan/common/docan_internal_def.h"
#include "diag/doip/include/api/doip_transport.h"
#include "diag/diag_server/include/common/diag_server_def.h"
#ifdef BUILD_SOMEIP_ENABLE
    #include "diag/dosomeip/api/dosomeip_transport.h"
#else
    #include "diag/dosomeip/common/dosomeip_def.h"
#endif

namespace hozon {
namespace netaos {
namespace diag {

const std::string Docan_Instance = "docan_instance";

struct docan_confirm
{
    uint16_t sa;
    uint16_t ta;
    uint32_t reqId;
    docan_result_t result;
    uint32_t length;
    std::vector<uint8_t> uds;
};

struct docan_indication
{
    uint16_t sa;
    uint16_t ta;
    uint32_t length;
    std::vector<uint8_t> uds;
};

class DocanListenerImpl : public DocanListener
{

public:
    DocanListenerImpl(std::function<void(docan_indication*)> indication_callback,
                      std::function<void(docan_confirm*)>    confirm_callback);

    virtual ~DocanListenerImpl();

public:
    virtual void OnUdsResponse(uint16_t sa, uint16_t ta, uint32_t reqId, docan_result_t result, const std::vector<uint8_t>& uds);

    virtual void OnUdsIndication(uint16_t ta, uint16_t sa, const std::vector<uint8_t>& uds);

    virtual void onServiceBind(const std::string& name);

    virtual void onServiceUnbind(const std::string& name);

private:
    DocanListenerImpl(const DocanListenerImpl &);
    DocanListenerImpl & operator = (const DocanListenerImpl &);

private:
    std::function<void(docan_indication*)> indication_callback;
    std::function<void(docan_confirm*)>  confirm_callback;
};

class DiagServerTransPortService {

public:
    static DiagServerTransPortService* getInstance();

    void Init();
    void DeInit();

    // docan
    bool DoCanStart(std::function<void(docan_indication*)> indication_callback,
                    std::function<void(docan_confirm*)> confirm_callback);
    bool DoCanStop();
    bool DoCanRequest(const std::string& instance, const DiagServerReqUdsMessage& udsMessage, const bool isDopipToDocan = false, const uint16_t doipAddress = 0x00, const bool someipChannel = false);
    int32_t GetDocanRequestId() {return request_id_docan_;}
    void DeleteDocanRequestId() {request_id_docan_ = -1;}

    // doip
    bool DoIPStart(std::function<void(doip_indication_t*)> indication_callback,
                   std::function<void(doip_confirm_t*)> confirm_callback,
                   std::function<void(doip_route_t*)> route_callback);
    bool DoIPStop();
    bool DoIPRequestByNode(const DiagServerReqUdsMessage& udsMessage);
    bool DoipRequestByEquip(const DiagServerReqUdsMessage& udsMessage, const bool someipChannel = false);
    void DoipReleaseByEquip(const DiagServerReqUdsMessage& udsMessage);
    bool GetDoipAddressByRequestId(const int32_t& requestId, uint16_t& address);
    void DeleteDoipAddressByRequestId(const int32_t& requestId);

    void DoIPLinkStatusCallBack(doip_netlink_status_t status, uint16_t address);
    bool IsDoipConnecting() {return current_link_address_list_.size() ? true : false;}

    // dosomeip
    bool DoSomeIPStart(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback, std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback);
    bool DoSomeIPStop();
    bool ReplyUdsOnSomeIp(const DoSomeIPRespUdsMessage& udsMsg, const Req_Channel& channel);

    Docan_Req_Channel GetDocanChannel();
    void setDocanChannelEmpty();

    Doip_Req_Channel GetDoipChannel();
    void setDoipChannelEmpty();

private:
    DiagServerTransPortService();
    DiagServerTransPortService(const DiagServerTransPortService &);
    DiagServerTransPortService & operator = (const DiagServerTransPortService &);
    void DoipSessionTimeout(void * data);
    void DocanSessionTimeout(void * data);


private:
    static DiagServerTransPortService* instance_;
    static std::mutex mtx_;

    DocanService* service_docan_;
    std::shared_ptr<DocanListener> listener_docan_impl_;
    int32_t request_id_docan_;

    DoIPTransport* service_doip_;
    std::unordered_map<int32_t, uint16_t> request_id_address_doip2docan_;
    std::vector<uint16_t> current_link_address_list_;

    Doip_Req_Channel doipReqChannel_;
    Docan_Req_Channel docanReqChannel_;
    int time_fd_doip_;
    int time_fd_docan_;
    std::unique_ptr<TimerManager> time_mgr_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_TRANSPORT_SERVICE_H
