
#include <thread>
#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <signal.h>
#include <memory>

#include "diag/docan/include/docan_service.h"
#include "diag/docan/include/docan_listener.h"
#include "diag/docan/common/docan_internal_def.h"
#include "diag/docan/log/docan_log.h"


using namespace hozon::netaos::diag;

uint8_t stopFlag = 0;

void SigHandler(int32_t signum)
{
    printf("--- docan process terminated, signum [%d] ---\n", signum);
    stopFlag = 1;
}

class DocanListenerImpl : public DocanListener
{
    using DocanListener::OnUdsResponse;
    using DocanListener::OnUdsIndication;
    
public:
    DocanListenerImpl()
        : DocanListener()
    {
    }
    virtual ~DocanListenerImpl()
    {
    }

public:
    virtual void OnUdsResponse(uint16_t canid, uint32_t reqId, docan_result_t result, const std::vector<uint8_t>& uds)
    {
        printf("OnUdsResponse from canid: %X, reqId: %d, result:%d, uds.size: %lu.\n", canid, reqId, result, uds.size());
    }

    virtual void OnUdsIndication(uint16_t canid, const std::vector<uint8_t>& uds)
    {
        printf("OnUdsIndication from canid: %X, uds.size: %lu.\n", canid, uds.size());
    }

    virtual void onServiceBind(const std::string& name)
    {
        printf("onServiceBind name: %s.\n", name.c_str());
    }

    virtual void onServiceUnbind(const std::string& name)
    {
        printf("onServiceUnbind name: %s.\n", name.c_str());
    }

private:
    DocanListenerImpl(const DocanListenerImpl &);
    DocanListenerImpl & operator = (const DocanListenerImpl &);
};


int32_t main(int32_t argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    DocanLogger::GetInstance().InitLogging("DOCAN",    // the id of application
        "docan test", // the log id of application
        DocanLogger::DocanLogLevelType::LOG_LEVEL_TRACE, //the log level of application
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        "./", //the log file directory, active when output log to file
        10, //the max number log file , active when output log to file
        20 //the max size of each  log file , active when output log to file
    );
    DocanLogger::GetInstance().CreateLogger("DOCAN");



    DocanService* service = new DocanService();
    std::shared_ptr<DocanListener> listener = std::shared_ptr<DocanListenerImpl>(new DocanListenerImpl());
    std::string who = "docan_test";

    service->Init();
    service->Start();

    service->registerListener(who, listener);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // docan_ecu_t ecu = FrontRadar;
    // docan_tatype_t tatype = docan_tatype_t(0);  // physic address
    uint16_t reqSa = 0x1062;
    uint16_t reqTa = 0x10C5;
    std::vector<uint8_t> uds_session        = { 0x10, 0x03 };
    std::vector<uint8_t> uds_access_seed    = { 0x27, 0x01 };
    std::vector<uint8_t> uds_access_key     = { 0x27, 0x02, 0x01, 0x02, 0x03, 0x04 };
    std::vector<uint8_t> uds_did_sn_read    = { 0x22, 0xF1, 0x8C };
    std::vector<uint8_t> uds_did_vin_write  = { 0x2E, 0xF1, 0x90, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x011, 0x012, 0x13, 0x14, 0x15, 0x16, 0x17 };
    std::vector<uint8_t> uds_routine_eol    = { 0x31, 0x01, 0x34, 0xF2 };
    std::vector<uint8_t> uds_routine_eol_res= { 0x31, 0x03, 0x34, 0xF2 };

    service->UdsRequest(who, reqSa, reqTa, uds_session);
    service->UdsRequest(who, reqSa, reqTa, uds_access_seed);
    service->UdsRequest(who, reqSa, reqTa, uds_access_key);
    service->UdsRequest(who, reqSa, reqTa, uds_did_sn_read);
    service->UdsRequest(who, reqSa, reqTa, uds_did_vin_write);
    service->UdsRequest(who, reqSa, reqTa, uds_routine_eol);
    service->UdsRequest(who, reqSa, reqTa, uds_routine_eol_res);

    while (!stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    service->unregisterListener(who);
    service->Stop();
    service->Deinit();
    delete service;
    service = nullptr;


    return 0;
}