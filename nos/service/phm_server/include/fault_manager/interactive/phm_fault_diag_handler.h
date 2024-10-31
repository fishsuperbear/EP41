
#ifndef PHM_FAULT_STUB_H
#define PHM_FAULT_STUB_H

#include <mutex>
#include "cm/include/method.h"
#include "idl/generated/diagPubSubTypes.h"
#include "phm_server/include/common/phm_server_def.h"
#include "phm_server/include/common/time_manager.h"

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace hozon::netaos::cm;

template <class T>
static T DataConversion(T t, std::ios_base & (*f)(std::ios_base&), int octal)
{
    std::ostringstream oss;
    int typesize = sizeof(t);
    if (1 == typesize) {
        uint8_t item = static_cast<uint8_t>(t);
        oss << f << static_cast<uint16_t>(item);
    }
    else {
        oss << f << t;
    }

    T result = static_cast<T>(std::strtoul(oss.str().c_str(), 0, octal));
    return result;
}

#define HEX_TO_DEC(type) DataConversion(type, std::hex, 10)   // example: 0x8004 -> 8004
#define DEC_TO_HEX(type) DataConversion(type, std::dec, 16)   // example: 8004 -> 0x8004

class DiagMessageMethodServer  : public Server<uds_data_method, uds_data_method> {
public:
    DiagMessageMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data) : Server(req_data, resp_data) {}
    virtual int32_t Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp);

    virtual ~DiagMessageMethodServer();
};

class DiagMessageMethodReceiver {
public:
    DiagMessageMethodReceiver();
    ~DiagMessageMethodReceiver();

    void Init();
    void DeInit();

private:
    std::shared_ptr<DiagMessageMethodServer> method_server_;
};

class DiagMessageHandler {

public:
    static DiagMessageHandler* getInstance();

    void Init();
    void DeInit();

    bool DealWithDiagMessage(const uint8_t sid, const uint8_t subid, std::vector<uint8_t>& messageData);

private:
    bool Start(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void StartReportFaultOccurOrRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void StartReportFaultOccurAndRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void StartRefreshFaultFile(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);

    bool Stop(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void StopReportFaultOccurOrRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void StopReportFaultOccurAndRecover(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);

    bool Result(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void ResultQueryCurrentFault(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void ResultQueryDtcByFault(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);
    void ResultQueryFaultByDtc(const std::vector<uint8_t>& reqData, std::vector<uint8_t>& respData);

    void FaultOccurOrRecover(void* data);
    void FaultOccurAndRecover(void* data);

private:
    DiagMessageHandler();
    DiagMessageHandler(const DiagMessageHandler &);
    DiagMessageHandler & operator = (const DiagMessageHandler &);

private:
    static std::mutex mtx_;
    static DiagMessageHandler* instance_;

    DiagMessageMethodReceiver* method_receiver_;
    std::shared_ptr<TimerManager> time_mgr_;

    Fault_t occur_or_recover_fault_;
    int fault_occur_or_recover_timer_fd_;
    bool fault_occur_or_recover_flag_;

    Fault_t occur_and_recover_fault_;
    int fault_occur_and_recover_timer_fd_;
    bool fault_occur_and_recover_flag_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_FAULT_STUB_H