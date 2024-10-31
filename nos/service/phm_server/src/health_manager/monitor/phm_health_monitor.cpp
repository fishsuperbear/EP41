#include "sm/include/state_client.h"
#include "em/include/proctypes.h"
#include "phm_server/include/health_manager/monitor/phm_health_gdb_monitor.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/health_manager/monitor/phm_health_monitor.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/common/phm_server_utils.h"
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"
#include <vector>
#include <set>

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace hozon::netaos::sm;
using namespace hozon::netaos::em;


static const std::string PROCESS_STATE_TERMINATING = "TERMINATING";
static const std::string PROCESS_STATE_TERMINATED = "TERMINATED";
static const std::string PROCESS_STATE_ABORTED = "ABORTED";


HealthMonitor::HealthMonitor()
: is_start(false)
, m_upPhmHealthResourcesMonitor(nullptr)
{
}

HealthMonitor::~HealthMonitor()
{
}

void
HealthMonitor::StartMonitor()
{
    PHMS_INFO << "HealthMonitor::StartMonitor";
    if (is_start) {
        PHMS_INFO << "HealthMonitor::StartMonitor task already start!";
        return;
    }

    is_start = true;
    work_thread_ = std::thread(&HealthMonitor::Run, this);

    m_upPhmHealthResourcesMonitor.reset(new PhmHealthGdbMonitor());
    m_upPhmHealthResourcesMonitor->CheckGdbDir();
    PHMS_INFO << "HealthMonitor::StartMonitor end";
}

void
HealthMonitor::StopMonitor()
{
    PHMS_INFO << "HealthMonitor::StopMonitor";
    is_start = false;

    if (work_thread_.joinable()) {
        PHMS_INFO << "HealthMonitor::StopMonitor work_thread_.join";
        work_thread_.join();
    }
}

void
HealthMonitor::SendProcFault(const std::string& procName, const uint8_t faultStatus)
{
    ProcInfo procinfo;
    if (!PHMServerConfig::getInstance()->GetProcInfoByName(procName, procinfo)) {
        PHMS_ERROR << "HealthMonitor::GetProcInfoByName not support procname: " << procName;
        return;
    }

    uint32_t fault_key = procinfo.faultId * 100 + procinfo.faultObj;
    FaultInfo faultInfo;
    if (!PHMServerConfig::getInstance()->GetFaultInfoByFault(fault_key, faultInfo)) {
        PHMS_ERROR << "HealthMonitor::GetFaultInfoByFault not support fault: " << fault_key;
        return;
    }

    Fault_t fault;
    fault.faultId = procinfo.faultId;
    fault.faultObj = procinfo.faultObj;
    fault.faultStatus = faultStatus;
    fault.faultOccurTime = PHMUtils::GetCurrentTime();
    fault.faultDomain = procName;
    FaultDispatcher::getInstance()->ReportFault(fault);

    return;
}

void
HealthMonitor::Run()
{
    PHMS_INFO << "HealthMonitor::Run";
    std::vector<std::string> process_state = {"IDLE", "STARTING", "RUNNING", "TERMINATING", "TERMINATED", "ABORTED"};
    std::set<std::string> faultProcessList;
    StateClient stateClient;
    const PhmConfigInfo& configInfo = PHMServerConfig::getInstance()->GetPhmConfigInfo();
    PHMS_INFO << "HealthMonitor::Run ProcCheckTime:" << configInfo.ProcCheckTime;

    while (is_start) {
        std::this_thread::sleep_for(std::chrono::milliseconds(configInfo.ProcCheckTime));
        std::vector<ProcessInfo> processInfos;
        int32_t res = stateClient.GetProcessInfo(processInfos);
        if (res != 0) {
            PHMS_WARN << "HealthMonitor::GetProcessInfo Failed!.";
            continue;
        }

        if (processInfos.empty()) {
            continue;
        }

        // PHMS_INFO << "HealthMonitor::Run processInfos size: " << processInfos.size();
        for(auto& item : processInfos) {
            // PHMS_INFO << "HealthMonitor::Run group: " << item.group
            //            << ", procname: " << item.procname.c_str()
            //            << ", procstate: " << process_state[static_cast<uint32_t>(item.procstate)].c_str();

            if ("" == item.procname) {
                continue;
            }

            auto processState = item.procstate;
            if (processState != ProcessState::TERMINATING
                && processState != ProcessState::TERMINATED
                && processState != ProcessState::ABORTED) {

                if (0 != faultProcessList.count(item.procname)) {
                    PHMS_INFO << "HealthMonitor::Run process " << item.procname.c_str() << " Recover!";
                    SendProcFault(item.procname, 0); // Recover
                    faultProcessList.erase(item.procname);
                }
            }
            else {
                if (0 == faultProcessList.count(item.procname)) {
                    PHMS_INFO << "HealthMonitor::Run process " << item.procname.c_str() << " Exeption!";
                    SendProcFault(item.procname, 1); // OCCUR
                    faultProcessList.insert(item.procname);

                    // get gdb infomation
                    if (m_upPhmHealthResourcesMonitor) {
                        m_upPhmHealthResourcesMonitor->StartCollectGdbInfo(item.procname);
                    }
                }
            }
        }
    }

}


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
