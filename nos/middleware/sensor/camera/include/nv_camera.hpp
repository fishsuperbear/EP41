#pragma once

#include <iostream>
#include <memory>
#include <functional>
#include <mutex>

#include "nv_camera_config.hpp"

namespace hozon {
namespace netaos {
namespace camera {

class NvCameraImpl;
class NvCamera
{
    public:
    static NvCamera& GetInstance();
    ~NvCamera();

    void Init(uint32_t sensor);
    void DeInit();

    using FrameAvailableCallback = std::function<void(void)>;
    int32_t RegisterProcess(FrameAvailableCallback callback, uint32_t sensor);

    int32_t GetImageData(std::string& image_data, uint32_t sensor);
    double GetImageTimeStamp(uint32_t sensor);
    uint32_t GetImageHeight(uint32_t sensor);
    uint32_t GetImageWidth(uint32_t sensor);
    uint32_t GetFrameID(uint32_t sensor);

private:
    NvCamera(/* args */);

    int32_t ParseConfigFromID(uint32_t sensor);
    SensorConfig _sensor_config;

    std::unique_ptr<NvCameraImpl> _pimpl;
    std::once_flag flag_init, flag_deinit;
};


}
}
}
