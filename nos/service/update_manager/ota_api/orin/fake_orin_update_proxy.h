#pragma once
#include <iostream>
#include <thread>

namespace Dsvpps {

    using UPDATE_RESULT = int32_t;
    // UPDATE_STATUS
    struct UPDATE_STATUS {
        int32_t status;
        int32_t progress;
    };

    using ParameterString = std::array<char,255>;
    // STD_RTYPE_E
    enum class STD_RTYPE_E:uint8_t {
        E_OK = 0x0,
        E_NOT_OK = 0x1,
    };

    // OTA_CURRENT_SLOT
    enum class OTA_CURRENT_SLOT:uint8_t {
        OTA_CURRENT_SLOT_A = 0x0,
        OTA_CURRENT_SLOT_B = 0x1,
    };

    using BaseTypeInt32 = int32_t;
    using PlaceHolder = int32_t;
    struct PpsPlaceHolderStruct{
        uint8_t placeHolder;
    };
} //namespace Dsvpps

int32_t updateStatus_{0};
int32_t virtualProgress {0};
bool updateProcess_{false};

void initPPS(){}

Dsvpps::ParameterString *g_payload_StartUpdateRequest2DSV = new Dsvpps::ParameterString();
Dsvpps::ParameterString &getStartUpdateRequest2DSV(){ return *g_payload_StartUpdateRequest2DSV; }
void sendStartUpdateRequest2DSVData(){}

static Dsvpps::PlaceHolder *g_payload_GetUpdateStatusRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getGetUpdateStatusRequest2DSV(){ return *g_payload_GetUpdateStatusRequest2DSV; }
void sendGetUpdateStatusRequest2DSVData() {
    if (!updateProcess_)
    {
        return;
    }
    updateStatus_ = 2;
    virtualProgress += 10;        
    if (virtualProgress >= 100)
    {
        virtualProgress = 100;
        updateProcess_ = false;
        updateStatus_ = 4;
    }
}

static Dsvpps::PlaceHolder *g_payload_GetVersionRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getGetVersionRequest2DSV(){ return *g_payload_GetVersionRequest2DSV; }
void sendGetVersionRequest2DSVData(){}

static Dsvpps::PlaceHolder *g_payload_PartitionSwitchRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getPartitionSwitchRequest2DSV(){ return *g_payload_PartitionSwitchRequest2DSV; }
void sendPartitionSwitchRequest2DSVData(){}

static Dsvpps::PlaceHolder *g_payload_GetCurrentPartitionRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getGetCurrentPartitionRequest2DSV(){ return *g_payload_GetCurrentPartitionRequest2DSV; }
void sendGetCurrentPartitionRequest2DSVData(){}

static Dsvpps::PlaceHolder *g_payload_RebootSystemRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getRebootSystemRequest2DSV(){ return *g_payload_RebootSystemRequest2DSV; }
void sendRebootSystemRequest2DSVData(){}

int32_t Get_Code_StartUpdate() { 
    updateProcess_ = true;
    std::cout << "wait ..." << std::endl;
    std::cout << "wait ..." << std::endl;
    std::cout << "wait ..." << std::endl;
    std::cout << "wait ..." << std::endl;
    std::cout << "wait ..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(50));
    return 0; 
}
int32_t Get_Code_SwitchSlot(){ return 0; }
std::string Get_Code_GetCurrentSlot(){ return ""; }
std::string Get_Code_GetVersion(){ return "DSV_VERSION"; }
int32_t Get_Code_Reboot(){ return 0; }

int32_t Get_UpdateState(){ return updateStatus_; }
int32_t Get_Update_Progress(){ return virtualProgress; }




