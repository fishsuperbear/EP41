// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef COMMON_HPP
#define COMMON_HPP

namespace hozon {
namespace netaos {
namespace desay { 

    constexpr uint32_t MAX_NUM_SENSORS= 16U;
    constexpr uint32_t MAX_OUTPUTS_PER_SENSOR = 4U;
    constexpr uint32_t MAX_NUM_PACKETS = 7U;
    constexpr uint32_t MAX_NUM_ELEMENTS = 8U;
    constexpr uint32_t NUM_IPC_CONSUMERS = 7U;
    constexpr uint32_t MAX_NUM_CONSUMERS = 8U;
    constexpr uint32_t MAX_WAIT_SYNCOBJ = MAX_NUM_CONSUMERS;
    constexpr uint32_t MAX_NUM_SYNCS = 8U;
    constexpr uint32_t MAX_QUERY_TIMEOUTS = 10U;
    constexpr int QUERY_TIMEOUT = 1000000; // usecs
    constexpr int QUERY_TIMEOUT_FOREVER = -1;
    constexpr uint32_t NVMEDIA_IMAGE_STATUS_TIMEOUT_MS = 100U;
    constexpr uint32_t DUMP_START_FRAME = 20U;
    constexpr uint32_t DUMP_END_FRAME = 120U;
    constexpr int64_t FENCE_FRAME_TIMEOUT_US = 100000U;
    constexpr const char *C2C_SRC_CHANNEL_PREFIX = "nvscic2c_pcie_s1_c5_";
    constexpr const char *C2C_DST_CHANNEL_PREFIX = "nvscic2c_pcie_s2_c5_";
    constexpr uint32_t MAX_NUM_WFD_PORTS = 2U;
    constexpr uint32_t MAX_NUM_WFD_PIPELINES = 2U;

    typedef enum
    {
        CommType_IntraProcess = 0,
        CommType_InterProcess,
        CommType_InterChip
    } CommType;

    typedef enum
    {
        EntityType_Producer = 0,
        EntityType_Consumer
    } EntityType;

    typedef enum
    {
        ProducerType_SIPL = 0,
        ProducerType_Display
    } ProducerType;

    // typedef enum
    // {
    //     ConsumerType_Enc = 0,
    //     ConsumerType_Cuda,
    //     ConsumerType_Stitch,
    //     ConsumerType_Display
    // } ConsumerType;

    typedef enum
    {
        QueueType_Mailbox = 0,
        QueueType_Fifo
    } QueueType;

    /* Names for the packet elements, should be 0~N */
    typedef enum
    {
        ELEMENT_TYPE_NV12_BL = 0,
        ELEMENT_TYPE_NV12_PL = 1,
        ELEMENT_TYPE_METADATA = 2,
        ELEMENT_TYPE_ABGR8888_PL = 3,
        ELEMENT_TYPE_ICP_RAW = 4
    } PacketElementType;

    /* Element information need to be set by user for producer and consumer */
    typedef struct
    {
        PacketElementType userType;
        bool isUsed = false;
        bool hasSibling = false;
    } ElementInfo;


    // Holds the I2C device bus number used to connect the deserializer with the SoC
#define DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER (0U)
// Holds the I2C device bus number used to connect the deserializer with the SoC(Orin)
#define DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER_CD (3U)
// Holds the deserializer I2C port number connected with the SoC
#define DESER_TO_SOC_I2C_PORT_NUMBER (0U)
#define DESER_TO_SOC_TX_PORT_NUMBER (UINT32_MAX)

}
}
}

#endif
