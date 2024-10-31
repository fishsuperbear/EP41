/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: remote diag extension switch control
*/

#include <thread>
#include <fstream>

#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/handler/remote_diag_handler.h"
#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/extension/remote_diag_switch_control.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

RemoteDiagSwitchControl* RemoteDiagSwitchControl::instance_ = nullptr;
std::mutex RemoteDiagSwitchControl::mtx_;

const std::string SSH_CONTROL_FILE_PATH = "/opt/usr/cfg/conf_app/remote_ctrl.cfg";

RemoteDiagSwitchControl::RemoteDiagSwitchControl()
{
}

RemoteDiagSwitchControl*
RemoteDiagSwitchControl::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new RemoteDiagSwitchControl();
        }
    }

    return instance_;
}

void
RemoteDiagSwitchControl::Init()
{
    DGR_INFO << "RemoteDiagSwitchControl::Init";
}

void
RemoteDiagSwitchControl::DeInit()
{
    DGR_INFO << "RemoteDiagSwitchControl::DeInit";
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }

    DGR_INFO << "RemoteDiagSwitchControl::DeInit finish!";
}

void
RemoteDiagSwitchControl::SwitchControl(const RemoteDiagSwitchControlInfo& switchInfo)
{
    DGR_INFO << "RemoteDiagSwitchControl::SwitchControl switchName: " << switchInfo.switchName;
    RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_SWITCH_CONTROL);
    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(switchInfo.sa.c_str(), 0, 16)), true);
    bool bControlResult = false;
    if ("SSH" == switchInfo.switchName) {
        bControlResult = SSHControl(switchInfo.control);
    }

    Json::Value respMessage;
    respMessage["SA"] = switchInfo.ta;
    respMessage["TA"] = switchInfo.sa;
    respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kSwitchControl];
    if (bControlResult) {
        respMessage["DATA"] = "Switch control successfully!";
    }
    else {
        respMessage["DATA"] = "Switch control abnormal!";
    }

    RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
    RemoteDiagHandler::getInstance()->SetCurrExtension(REMOTE_DIAG_EXTENSION::REMOTE_DIAG_DEFAULT);
    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(static_cast<uint16_t>(std::strtoul(switchInfo.sa.c_str(), 0, 16)), false);
}

bool
RemoteDiagSwitchControl::SSHControl(std::string controlType)
{
    DGR_INFO << "RemoteDiagSwitchControl::SSHControl controlType: " << controlType;
    bool bControlResult = false;
    if ("open" == controlType) {
        if (access(SSH_CONTROL_FILE_PATH.c_str(), F_OK) == 0) {
            bControlResult = true;
        }
        else {
            std::ofstream ofs(SSH_CONTROL_FILE_PATH);
            ofs.close();
            if (access(SSH_CONTROL_FILE_PATH.c_str(), F_OK) == 0) {
                bControlResult = true;
            }
        }
    }
    else if ("close" == controlType) {
        if (access(SSH_CONTROL_FILE_PATH.c_str(), F_OK) != 0) {
            bControlResult = true;
        }
        else {
            int ret = remove(SSH_CONTROL_FILE_PATH.c_str());
            if (0 == ret) {
                bControlResult = true;
            }
        }
    }
    else {
        DGR_WARN << "RemoteDiagSwitchControl::SSHControl invalid controlType: " << controlType;
    }

    return bControlResult;
}

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon