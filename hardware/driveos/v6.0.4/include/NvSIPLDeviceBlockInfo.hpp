/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVSIPLDEVICEBLOCKINFO_HPP
#define NVSIPLDEVICEBLOCKINFO_HPP

#include "NvSIPLCapStructs.h"

#include <string>
#include <memory>
#include <vector>
#include <cmath>

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: DeviceBlock Information - @ref NvSIPLDevBlkInfo </b>
 *
 */

namespace nvsipl
{
/** @defgroup NvSIPLDevBlkInfo NvSIPL DeviceBlock Information
 *
 *  @brief Describes information about devices supported by SIPL Device Block.
 *
 *  @ingroup NvSIPLCamera_API
 */

 /** @addtogroup NvSIPLDevBlkInfo
 * @{
 */

/** @brief Indicates the maximum number of device blocks per platform. */
#if !NV_IS_SAFETY
static const uint32_t MAX_DEVICEBLOCKS_PER_PLATFORM = 6U;
#else
static const uint32_t MAX_DEVICEBLOCKS_PER_PLATFORM = 4U;
#endif

/** @brief Indicates the maximum number of camera modules per device block. */
static const uint32_t MAX_CAMERAMODULES_PER_BLOCK = 4U;

/** @brief Indicates the maximum number of camera modules per platform. */
static const uint32_t MAX_CAMERAMODULES_PER_PLATFORM = MAX_DEVICEBLOCKS_PER_PLATFORM * MAX_CAMERAMODULES_PER_BLOCK;

/** @brief Indicates the maximum number of sensors per platform. */
static const uint32_t MAX_SENSORS_PER_PLATFORM = MAX_CAMERAMODULES_PER_PLATFORM;

/** @brief Indicates the maximum number of CSI lane configurations */
static const std::uint32_t MAX_CSI_LANE_CONFIGURATION = 2U;

/** @brief Indicates the index for CSI 2 lanes */
static const std::uint32_t X2_CSI_LANE_CONFIGURATION = 0U;

/** @brief Indicates the index for CSI 4 lanes */
static const std::uint32_t X4_CSI_LANE_CONFIGURATION = 1U;

/** @brief Defines the image sensor information. */
struct SensorInfo
{
    /** @brief Defines the image resolution.
     * @anon_struct
     */
    struct Resolution
    {
        /** @brief < Holds the width in pixels in the range
         * from NVSIPL_CAP_MIN_IMAGE_WIDTH to NVSIPL_CAP_MAX_IMAGE_WIDTH.
         * @anon_struct_member
         */
        uint32_t width = 0U;

        /** @brief < Holds the height in pixels in the range
         * from NVSIPL_CAP_MIN_IMAGE_HEIGHT to NVSIPL_CAP_MAX_IMAGE_HEIGHT.
         * @anon_struct_member
         */
        uint32_t height = 0U;
    };

    /** @brief Defines the information of a virtual channel/single exposure.
     * @anon_struct{1}
     */
    struct VirtualChannelInfo
    {
        /**
         * @brief
         *
         * Holds the Bayer color filter array order of the sensor. NVIDIA
         * recommends setting this element with the
         * NVSIPL_PIXEL_ORDER_* family of macros, e.g.
         * \ref NVSIPL_PIXEL_ORDER_ALPHA.
         * @anon_struct_member{1}
         */
        uint32_t cfa = 0U;
        /** @brief Holds the number of top embedded lines.
         * @anon_struct_member{1}
         */
        uint32_t embeddedTopLines = UINT32_MAX;
        /** @brief Holds the number of bottom embedded lines.
         * @anon_struct_member{1}
         */
        uint32_t embeddedBottomLines = UINT32_MAX;
        /** @brief Holds the input format
         * @anon_struct_member{1}
         */
        NvSiplCapInputFormatType inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12;
        /** @brief Holds the @ref Resolution of the captured frame.
         * @anon_struct_member{1}
         */
        Resolution resolution;
        /** @brief Holds the average number of frames per second
         *  in the range from NVSIPL_CAP_MIN_FRAME_RATE to NVSIPL_CAP_MAX_FRAME_RATE.
         * @anon_struct_member{1}
         */
        float_t fps = 0.0F;
        /** @brief Indicates whether the embedded data is coming in CSI packet with different data. The default value is <em>false</em>.
         * @anon_struct_member{1}
         */
        bool isEmbeddedDataTypeEnabled = false;
    };

    /** @brief Holds the identification of the pipeline index in the platform configuration. */
    uint32_t id = UINT32_MAX;
    /** @brief Holds the name of the image sensor, for example, "AR0231". */
    std::string name = "";
#if !NV_IS_SAFETY
    /** Holds the description of the image sensor. */
    std::string description = "";
#endif // !NV_IS_SAFETY
    /** @brief Holds the native I2C address of the image sensor. */
    uint8_t i2cAddress = static_cast<uint8_t>UINT8_MAX;
    /** @brief Holds virtual channel information. */
    VirtualChannelInfo vcInfo;
    /** @brief Holds a flag which indicates whether trigger mode is enabled. The
     default value is <em>false</em>. */
    bool isTriggerModeEnabled = false;
    /** @brief Holds CDAC Error GPIO indices for the sensor. */
    std::vector<uint32_t> errGpios;
    /** @brief Holds a flag which indicates whether CDI (Camera Device Interface)
     is using the new version 2 API for device. The default value is <em>false</em>
     for version 1 API. The flag is only used for non-Safety build.
     For Safety build, CDI is always using version 2 API */
    bool useCDIv2API = false;
#if !NV_IS_SAFETY
    /** Holds a flag which indicates whether the sensor requires configuration
     in Test Pattern Generator (TPG) mode. The default value is
     <em>false</em>. */
    bool isTPGEnabled = false;
    /** Holds the Test Pattern Generator (TPG) pattern mode. */
    uint32_t patternMode = 0;
#endif // !NV_IS_SAFETY
};

/** @brief Defines the EEPROM information. */
struct EEPROMInfo
{
    /** @brief Holds the name of the EEPROM, for example, "N24C64". */
    std::string name = "";
#if !NV_IS_SAFETY
    /** Holds the description of the EEPROM. */
    std::string description = "";
#endif //!NV_IS_SAFETY
    /** @brief Holds the native I2C address. */
    uint8_t i2cAddress = static_cast<uint8_t>UINT8_MAX;
    /** @brief Holds a flag which indicates whether CDI (Camera Device Interface)
     is using the new version 2 API for device. The default value is <em>false</em>
     for version 1 API. The flag is only used for non-Safety build.
     For Safety build, CDI is always using version 2 API */
    bool useCDIv2API = false;
};

/** @brief Defines GPIO mapping from the serializer to the deserializer */
struct SerdesGPIOPinMap {
    uint8_t sourceGpio;
    uint8_t destGpio;
};

/** @brief Defines the serializer information. */
struct SerInfo
{
    /** @brief Holds the name of the serializer, for example, "MAX96705". */
    std::string name = "";
#if !NV_IS_SAFETY
    /** Holds the description of the serializer. */
    std::string description = "";
#endif // !NV_IS_SAFETY
    /** @brief Holds the native I2C address. */
    uint8_t i2cAddress = static_cast<uint8_t>UINT8_MAX;
#if !NV_IS_SAFETY
    /** Holds long cable support */
    bool longCable = false;
#endif // !NV_IS_SAFETY
    /** @brief Holds CDAC Error GPIO indices for the serializer. */
    std::vector<uint32_t> errGpios;
    /** @brief Holds a flag which indicates whether CDI (Camera Device Interface)
     is using the new version 2 API for device. The default value is <em>false</em>
     for version 1 API. The flag is only used for non-Safety build.
     For Safety build, CDI is always using version 2 API */
    bool useCDIv2API = false;
    /** @brief Holds the information about GPIO mapping from the serializer to the deserializer
     Holds two numbers as one set GPIO mapping from the serializer to the deserializer
     the size of the vector should be even */
    std::vector<SerdesGPIOPinMap> serdesGPIOPinMappings;
};

/** @brief Defines information for the camera module.
 *
 * A camera module is a physical grouping of a serializer,
 * image sensor(s), and associated EEPROM(s). */
struct CameraModuleInfo
{
    /** @brief Holds the name of the camera module, for example, "SF3324". */
    std::string name = "";
#if !NV_IS_SAFETY
    /** Holds the description of the camera module. */
    std::string description = "";
#endif // !NV_IS_SAFETY
    /** @brief Holds  the index of the deserializer link to which this module is
     connected. */
    uint32_t linkIndex = UINT32_MAX;
    /** @brief Holds the @ref SerInfo of the serializer. */
    SerInfo serInfo;
    /** @brief Holds EEPROM support */
    bool isEEPROMSupported = false;
    /** @brief Holds the information about EEPROM device
     in a camera module. */
    EEPROMInfo eepromInfo;
    /** @brief Holds the information about the sensor in a camera module. */
    SensorInfo sensorInfo;
};

/** @brief Defines the deserializer information. */
struct DeserInfo
{
    /** @brief Holds the name of the deserializer, for example, "MAX96712". */
    std::string name = "";
#if !NV_IS_SAFETY
    /** Holds the description of the deserializer. */
    std::string description = "";
#endif // !NV_IS_SAFETY
    /** @brief Holds the native I2C address of the deserializer. */
    uint8_t i2cAddress = static_cast<uint8_t>UINT8_MAX;
    /** @brief Holds CDAC Error GPIO indices for the deserializer. */
    std::vector<uint32_t> errGpios;
    /** @brief Holds a flag which indicates whether CDI (Camera Device Interface)
     is using the new version 2 API for device. The default value is <em>false</em>
     for version 1 API. The flag is only used for non-Safety build.
     For Safety build, CDI is always using version 2 API */
    bool useCDIv2API = false;
    /* @brief Holds flag to indicate that the deser needs to run a reset all sequence at
     startup */
    bool resetAll = false;
};

/**
 * @brief  Defines the DeviceBlock information.
 *
 * A DeviceBlock represents a grouping of a deserializer (which is connected
 * to the SoC's CSI interface) and the camera modules connected
 * to the links of the deserializer. */
struct DeviceBlockInfo
{
    /** Holds the @ref NvSiplCapInterfaceType that specifies the CSI port of
     the SoC to which the deserializer is connected. */
    NvSiplCapInterfaceType csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A;
    /** Holds the @ref NvSiplCapCsiPhyMode Phy mode. */
    NvSiplCapCsiPhyMode phyMode = NVSIPL_CAP_CSI_DPHY_MODE;
    /** @brief Holds the I2C device bus number used to connect the deserializer with
     the SoC. */
    uint32_t i2cDevice = UINT32_MAX;
    /** @brief Holds the @ref DeserInfo deserializer information. */
    DeserInfo deserInfo;
    /** @brief Holds the number of camera modules connected to the deserializer.
     * This value must be less than or equal to
     * @ref MAX_CAMERAMODULES_PER_BLOCK. */
    uint32_t numCameraModules = 0U;
    /** @brief Holds an array of information about each camera module
     in the device block. */
    CameraModuleInfo cameraModuleInfoList[MAX_CAMERAMODULES_PER_BLOCK];
    /** @brief Holds the deserializer I2C port number connected with the SoC. */
    std::uint32_t desI2CPort = UINT32_MAX;
    /** @brief Holds the deserializer Tx port number connected with the SoC. */
    std::uint32_t desTxPort = UINT32_MAX;
    /** @brief Holds the power port */
    std::uint32_t pwrPort = UINT32_MAX;
    /** @brief Holds the deserializer's data rate in DPHY mode(kHz) */
    std::uint32_t dphyRate[MAX_CSI_LANE_CONFIGURATION] = {0U};
    /** @brief Holds the deserializer's data rate in CPHY mode(ksps) */
    std::uint32_t cphyRate[MAX_CSI_LANE_CONFIGURATION] = {0U};
    /** Holds a flag which indicates whether simulator mode has been enabled.
     Used for the ISP reprocessing use case to simulate the presence of a
     device block. */
    bool isSimulatorModeEnabled = false;
    /** Holds a flag which indicates whether passive mode must be enabled.
     Used when an NVIDIA DRIVE&trade; AGX SoC connected to the deserializer
     does not have an I2C connection to control it. */
    bool isPassiveModeEnabled = false;
    /** @brief Holds a flag which indicates whether group initialization is enabled. */
    bool isGroupInitProg = false;
    /** @brief Holds CDAC GPIO indices for the Device Block. (Non-error monitoring.) */
    std::vector<uint32_t> gpios;
#if !NV_IS_SAFETY
    /** Holds a flag which indicates whether power control is disabled on the
     platform. */
    bool isPwrCtrlDisabled = false;
    /** Holds long cable support */
    bool longCables[MAX_CAMERAMODULES_PER_BLOCK] = {false, false, false, false};
#endif // !NV_IS_SAFETY
    /* @brief Reset all sequence is needed when starting deserializer. */
    bool resetAll;
};

/** @} */

} // namespace nvsipl

#endif //NVSIPLDEVICEBLOCKINFO_HPP
