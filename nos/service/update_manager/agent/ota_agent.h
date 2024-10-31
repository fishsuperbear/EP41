#ifndef OTA_AGENT_H
#define OTA_AGENT_H

#include <stdint.h>
#include <mutex>

#include "interface_update_req_dispatcher.h"

namespace hozon {
namespace netaos {
namespace update {


class OTAAgent {
public:

    static OTAAgent* Instance();

    void Init();
    void Deinit();

    // mdc
    bool Activate();
    bool Finish();
    bool GetActivationProgress(uint8_t& progress, std::string& message);

    // orin
    bool QueryStatus(std::string& updateStatus, uint8_t& progress);
    bool SwitchSlot();
    bool GetCurrentSlot(std::string& currentSlot);
    bool Reboot();

    // orin && mdc
    bool Update(const std::string& packageName);
    bool Query(std::string& updateStatus);
    bool GetUpdateProgress(uint8_t& progress, std::string& message);
    bool GetVersionInfo(std::string& mdcVersion);

    bool HardReboot();
private:
    OTAAgent();
    ~OTAAgent();
    OTAAgent(const OTAAgent &);
    OTAAgent & operator = (const OTAAgent &);

    static std::mutex m_mtx;
    static OTAAgent* m_pInstance;

    std::unique_ptr<InterfaceUpdateReqDispatcher> update_interface_req_dispatcher_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // OTA_AGENT_H
