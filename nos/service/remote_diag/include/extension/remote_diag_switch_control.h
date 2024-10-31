#ifndef REMOTE_DIAG_SWITCH_CONTROL_H
#define REMOTE_DIAG_SWITCH_CONTROL_H

#include <mutex>
#include <iostream>

#include "remote_diag/include/common/remote_diag_def.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

class RemoteDiagSwitchControl {

public:
    static RemoteDiagSwitchControl* getInstance();

    void Init();
    void DeInit();

    void SwitchControl(const RemoteDiagSwitchControlInfo& switchInfo);

private:
    RemoteDiagSwitchControl();
    RemoteDiagSwitchControl(const RemoteDiagSwitchControl &);
    RemoteDiagSwitchControl & operator = (const RemoteDiagSwitchControl &);

private:
    bool SSHControl(std::string control);

private:
    static RemoteDiagSwitchControl* instance_;
    static std::mutex mtx_;
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // #define REMOTE_DIAG_SWITCH_CONTROL_H