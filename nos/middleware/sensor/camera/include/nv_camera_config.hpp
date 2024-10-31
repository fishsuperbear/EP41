#pragma once

#include <iostream>
#include <memory>
#include <functional>
#include <vector>

namespace hozon {
namespace netaos {
namespace camera {

#define NITO_PATH "/usr/share/camera/"

// Sensor Config
#define X8B40 "XPC_F120_OX08B40_MAX96717_CPHY_x4"
#define X031  "XPC_ISX031_YUV422_TEST_DPHY_x4"
#define X3F  "XPC_S100_OX03F10_MAX96717F_A_CPHY_x4"
#define MULTI  "XPC_MULTIPLE_CAMERA"

#define X8B40_F120_NITO "CA_F120_OX08B40_MAX96717"
#define X8B40_F30_NITO "CA_F30_OX08B40_MAX96717"
#define X3F_NITO  "CA_S100_OX03F10_MAX96717F_A"

#define SENSOR_0X8B40 "OX08B40"
#define SENSOR_ISX031  "isx031"
#define SENSOR_0X03F  "OX03F10"

const std::vector<std::uint32_t> x8b40_mask_list = {
    0x0000, 0x0010, 0x0000, 0x0
};

const std::vector<std::uint32_t> x031_mask_List = {
    0x0, 0x0, 0x0001, 0x0
};

const std::vector<std::uint32_t> x3f_side_mask_List = {
    0x0000, 0x0001, 0x0, 0x0
};

const std::vector<std::uint32_t> multi_mask_List = {
    0x1011, 0x1111, 0x1111, 0x0
    // 0x1011, 0x0001, 0x1111, 0x0
    // 0x1000, 0x0, 0x0, 0x0
};

enum ThreadIndex {
    THREAD_INDEX_ICP = 0U,
    THREAD_INDEX_ISP0,
    THREAD_INDEX_ISP1,
    THREAD_INDEX_ISP2,
    THREAD_INDEX_EVENT,
    THREAD_INDEX_COUNT
};

struct PipelineCfg {
    std::string sensor_name;
    bool captureOutputRequested = false;
    bool isp0OutputRequested = false;
    bool isp1OutputRequested = false;
    bool isp2OutputRequested = false;
};

struct SensorConfig {
    std::string platform_name;
    struct PipelineCfg pipeline[3];
    std::vector<uint32_t> vMasks;
};

}
}
}
