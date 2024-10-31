
#ifndef DIAG_AGENT_HANDLER_IMPL_H
#define DIAG_AGENT_HANDLER_IMPL_H

#include <mutex>
#include <functional>
#include <unordered_map>
#include "cm/include/method.h"
#include "idl/generated/diagPubSubTypes.h"
#include "diag/diag_agent/include/service/diag_agent_data_identifier.h"
#include "diag/diag_agent/include/service/diag_agent_routine_control.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent{

using namespace hozon::netaos::cm;

enum DiagRidStatus {
    kDefault,
    kStarted,
    kStopped
};

struct DiagMessageInfo {
    uint8_t sid;
    uint8_t subid;
    std::vector<uint8_t> messageData;
};

class DiagAgentMethodServer  : public Server<uds_data_method, uds_data_method> {
public:
    DiagAgentMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data,
                            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data,
                            std::function<bool(DiagMessageInfo*)> dealDiagMessage)
                            : Server(req_data, resp_data)
                            , deal_diag_message_(dealDiagMessage)
                            {
                            }
    virtual ~DiagAgentMethodServer();

    virtual int32_t Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp);

private:
    std::function<bool(DiagMessageInfo*)> deal_diag_message_;
};

class DiagAgentMethodReceiver {
public:
    DiagAgentMethodReceiver();
    ~DiagAgentMethodReceiver();

    void Init(const std::string& processName, std::function<bool(DiagMessageInfo*)> dealDiagMessage);
    void DeInit();

private:
    std::shared_ptr<DiagAgentMethodServer> method_server_;
};

class DiagAgentHandlerImpl {

public:
    DiagAgentHandlerImpl();
    ~DiagAgentHandlerImpl();

    DiagAgentInitResultCode Init(const std::string& configPath,
                 std::shared_ptr<DiagAgentDataIdentifier> dataIdentifier,
                 std::shared_ptr<DiagAgentRoutineControl> routineControl);
    void DeInit();

    bool DealWithDiagMessage(DiagMessageInfo* messageInfo);

private:
    char* GetJsonAll(const char *fname);
    bool LoadDiagAgentConfig(const std::string& configPath);
    bool ReadDataIdentifier(std::vector<uint8_t>& messageData);
    bool WriteDataIdentifier(std::vector<uint8_t>& messageData);
    bool RoutineControl(const uint8_t subid, std::vector<uint8_t>& messageData);

private:
    DiagAgentHandlerImpl(const DiagAgentHandlerImpl &);
    DiagAgentHandlerImpl & operator = (const DiagAgentHandlerImpl &);

private:
    std::shared_ptr<DiagAgentDataIdentifier> data_identifier_;
    std::shared_ptr<DiagAgentRoutineControl> routine_control_;

    DiagAgentMethodReceiver* method_receiver_;

    std::string process_name_;
    std::vector<DiagAgentDidDataInfo> read_did_list_;
    std::vector<DiagAgentDidDataInfo> write_did_list_;
    std::vector<DiagAgentRidDataInfo> rid_list_;

    std::unordered_map<uint16_t, DiagRidStatus> rid_status_map_;
};

}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_AGENT_HANDLER_IMPL_H