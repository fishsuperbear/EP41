#pragma once

#include <mutex>
#include <memory>

#include "log_server/common/log_server_def.h"
#include "phm/include/phm_client.h"

using namespace hozon::netaos::phm;
namespace hozon {
namespace netaos {
namespace logserver {

class logServerFaultHandler {

public:
    static logServerFaultHandler* getInstance();

    void Init();
    void DeInit();
    bool ReportFault(const LogServerSendFaultInfo& faultInfo);

private:
    logServerFaultHandler();
    logServerFaultHandler(const logServerFaultHandler &);
    logServerFaultHandler & operator = (const logServerFaultHandler &);

private:
    static std::mutex mtx_;
    static logServerFaultHandler* instance_;
    
    std::unique_ptr<PHMClient> phm_client_;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
