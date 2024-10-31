// Generated automatically from impltypes.csv by giant.py.
// Powered by python 3.8.
#ifndef _DSV_PPS_IF_STRUCT_H_
#define _DSV_PPS_IF_STRUCT_H_

#include <stdio.h>
#include <array>

#include <stdint.h>

namespace Dsvpps {

    enum eTopic {
        E_PowerMgrMsg2HZ = 1,
        E_PowerMgrMsg2DSV = 2,
        E_StartUpdateResponse2HZ = 3,
        E_GetUpdateStatusResponse2HZ = 4,
        E_GetVersionResponse2HZ = 5,
        E_PartitionSwitchResponse2HZ = 6,
        E_GetCurrentPartitionResponse2HZ = 7,
        E_RebootSystemResponse2HZ = 8,
        E_StartUpdateRequest2DSV = 9,
        E_GetUpdateStatusRequest2DSV = 10,
        E_GetVersionRequest2DSV = 11,
        E_PartitionSwitchRequest2DSV = 12,
        E_GetCurrentPartitionRequest2DSV = 13,
        E_RebootSystemRequest2DSV = 14,
    };

    using DsvPowerMgrMsg_Array = std::array<uint8_t,128>;
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

#endif // _DSV_PPS_IF_STRUCT_H_
