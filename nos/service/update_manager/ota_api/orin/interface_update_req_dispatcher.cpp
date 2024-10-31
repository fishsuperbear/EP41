/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: update interface req dispatcher
*/

#include "interface_update_req_dispatcher.h"
#include "update_manager/log/update_manager_logger.h"
#ifdef BUILD_FOR_ORIN
    #include "desay/include/pps_update_manager.h"
#endif

#ifdef BUILD_FOR_X86
    #include "fake_orin_update_proxy.h"
#endif

namespace hozon {
namespace netaos {
namespace update {


InterfaceUpdateReqDispatcher::InterfaceUpdateReqDispatcher()
{
}

InterfaceUpdateReqDispatcher::~InterfaceUpdateReqDispatcher()
{
}

int32_t
InterfaceUpdateReqDispatcher::Init()
{
    initPPS();
    return 0;
}

void
InterfaceUpdateReqDispatcher::Deinit()
{
}

int32_t
InterfaceUpdateReqDispatcher::Update(const std::string& packageName)
{
    UPDATE_LOG_D("Update interface Update packageName: %s", packageName.c_str());
    Dsvpps::ParameterString &result =  getStartUpdateRequest2DSV();
    memcpy(&result, packageName.c_str(), strlen(packageName.c_str()));
    
    sendStartUpdateRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    return Get_Code_StartUpdate();
}

int32_t
InterfaceUpdateReqDispatcher::Query(std::string& updateStatus, uint8_t& progress)
{
    sendGetUpdateStatusRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto res1 = Get_UpdateState();
    switch (res1)
    {
    case 0:
        updateStatus = "IDLE";
        break;
    case 1:
    case 2:
        updateStatus = "UPDATING";
        break;
    case 3:
        updateStatus = "UPDATE_FAILED";
        break;
    case 4:
        updateStatus = "UPDATE_SUCCESS";
        break;
    default:
        updateStatus = "DEFAULT";
        break;
    }
    progress = static_cast<uint8_t>(Get_Update_Progress());
    return 0;
}

int32_t
InterfaceUpdateReqDispatcher::QueryUpdateProgress(uint8_t& progress)
{
    sendGetUpdateStatusRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    progress = static_cast<uint8_t>(Get_Update_Progress());
    return 0;
}

int32_t
InterfaceUpdateReqDispatcher::QueryUpdateStatus(std::string& updateStatus)
{
    sendGetUpdateStatusRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto res1 = Get_UpdateState();
    switch (res1)
    {
    case 0:
        updateStatus = "IDLE";
        break;
    case 1:
    case 2:
        updateStatus = "UPDATING";
        break;
    case 3:
        updateStatus = "UPDATE_FAILED";
        break;
    case 4:
        updateStatus = "UPDATE_SUCCESS";
        break;
    default:
        updateStatus = "DEFAULT";
        break;
    }
    return 0;
}

int32_t 
InterfaceUpdateReqDispatcher::SwitchSlot()
{
    sendPartitionSwitchRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return Get_Code_SwitchSlot();
}

int32_t 
InterfaceUpdateReqDispatcher::GetCurrentSlot(std::string& currentSlot)
{
    sendGetCurrentPartitionRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    currentSlot = Get_Code_GetCurrentSlot();
    if (currentSlot == "") {
        return 1;
    }
    return 0;
}

int32_t 
InterfaceUpdateReqDispatcher::Reboot()
{
    sendRebootSystemRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return Get_Code_Reboot();
}

int32_t 
InterfaceUpdateReqDispatcher::GetVersionInfo(std::string& version) 
{
    sendGetVersionRequest2DSVData();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    version = Get_Code_GetVersion();
    return 0;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon