/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _Sensor_CustomInterface_HPP_
#define _Sensor_CustomInterface_HPP_

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "NvSIPLCommon.hpp"

namespace nvsipl
{
// This is version 1 UUID obtained using https://www.uuidgenerator.net/
// This will be used to uniquely identify this interface
// The client can use this ID to validate the correct interface before use
const UUID SENSOR_CUSTOM_INTERFACE_ID(0xc6490294U, 0xfe57U, 0x4bf1U, 0xbe32U,
	                                    0x39U, 0xf8U, 0x51U, 0x4eU, 0x8bU, 0x9bU);

class Sensor_CustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return SENSOR_CUSTOM_INTERFACE_ID;
    }

    const UUID& getInstanceInterfaceID() {
        return SENSOR_CUSTOM_INTERFACE_ID;
    }

    virtual SIPLStatus IWriteRegisterSensor(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;

    virtual SIPLStatus IReadRegisterSensor(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;

    virtual SIPLStatus CheckModuleStatus(bool *linkdone) = 0;

    virtual SIPLStatus SetPowerControl() = 0;
protected:
    ~Sensor_CustomInterface() = default;
};

const UUID SER_CUSTOM_INTERFACE_ID(0xc5394272U, 0xfd56U, 0x3bf3U, 0xce61U,
	                                    0x69U, 0xf5U, 0x30U, 0x23U, 0x74U, 0x8aU);

class SER_CustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return SER_CUSTOM_INTERFACE_ID;
    }

    const UUID& getInstanceInterfaceID() {
        return SER_CUSTOM_INTERFACE_ID;
    }

    virtual SIPLStatus IWriteRegisterSER(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;

    virtual SIPLStatus IReadRegisterSER(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;
protected:
    ~SER_CustomInterface() = default;
};

const UUID DeSER_CUSTOM_INTERFACE_ID(0xc7304372U, 0xbb59U, 0x1bf2U, 0xba51U,
	                                    0x49U, 0xa4U, 0x28U, 0x19U, 0x18U, 0x32U);

class DeSER_CustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return DeSER_CUSTOM_INTERFACE_ID;
    }

    const UUID& getInstanceInterfaceID() {
        return DeSER_CUSTOM_INTERFACE_ID;
    }

    virtual SIPLStatus IWriteRegisterDeSER(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;

    virtual SIPLStatus IReadRegisterDeSER(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;
protected:
    ~DeSER_CustomInterface() = default;
};

const UUID EEPROM__CUSTOM_INTERFACE_ID(0x25d6fd45U, 0xa2a7U, 0x4de8U, 0xaa10U,
	                                    0x48U, 0xd5U, 0x3eU, 0x7dU, 0x74U, 0xb2U);
class EEPROM_CustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return EEPROM__CUSTOM_INTERFACE_ID;
    }

    const UUID& getInstanceInterfaceID() {
        return EEPROM__CUSTOM_INTERFACE_ID;
    }

    virtual SIPLStatus IWriteRegisterEEPROM(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;

    virtual SIPLStatus IReadRegisterEEPROM(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;
protected:
    ~EEPROM_CustomInterface() = default;
};

const UUID CAMPWR__CUSTOM_INTERFACE_ID(0x22852122U, 0xbd77U, 0x11edU, 0xa330U,
	                                    0xbbU, 0xc1U, 0x51U, 0x46U, 0xdbU, 0xf5U);
class CAMPWR_CustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return CAMPWR__CUSTOM_INTERFACE_ID;
    }

    const UUID& getInstanceInterfaceID() {
        return CAMPWR__CUSTOM_INTERFACE_ID;
    }

    virtual SIPLStatus IWriteRegisterCAMPWR(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;

    virtual SIPLStatus IReadRegisterCAMPWR(uint16_t registerAddr, uint8_t* data, uint16_t length) = 0;
protected:
    ~CAMPWR_CustomInterface() = default;
};

} // end of namespace nvsipl
#endif // _Sensor_CustomInterface_HPP_
