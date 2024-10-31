#include <iostream>

#include "NvSIPLCommon.hpp"

#include "nv_camera.hpp"
#include "nv_camera_impl.hpp"

namespace hozon {
namespace netaos {
namespace camera {

NvCamera& NvCamera::GetInstance() {
    static NvCamera instance;
    return instance;
}

NvCamera::NvCamera():
    _pimpl(new NvCameraImpl()) {

}

NvCamera::~NvCamera() {

}

int32_t NvCamera::ParseConfigFromID(uint32_t sensor) {
    // Sensor Config
    _sensor_config.platform_name = MULTI;
    _sensor_config.pipeline[0].sensor_name = SENSOR_ISX031;
    _sensor_config.pipeline[0].captureOutputRequested = true;
    _sensor_config.pipeline[0].isp0OutputRequested = false;
    _sensor_config.pipeline[0].isp1OutputRequested = false;
    _sensor_config.pipeline[0].isp2OutputRequested = false;

    _sensor_config.pipeline[1].sensor_name = SENSOR_0X8B40;
    _sensor_config.pipeline[1].captureOutputRequested = false;
    _sensor_config.pipeline[1].isp0OutputRequested = true;
    _sensor_config.pipeline[1].isp1OutputRequested = false;
    _sensor_config.pipeline[1].isp2OutputRequested = false;

    _sensor_config.pipeline[2].sensor_name = SENSOR_0X03F;
    _sensor_config.pipeline[2].captureOutputRequested = false;
    _sensor_config.pipeline[2].isp0OutputRequested = true;
    _sensor_config.pipeline[2].isp1OutputRequested = false;
    _sensor_config.pipeline[2].isp2OutputRequested = false;

    _sensor_config.vMasks = multi_mask_List;

    return 0;
}

void NvCamera::Init(uint32_t sensor) {
    std::call_once(flag_init, [&](){ 
        ParseConfigFromID(sensor);
        _pimpl->Init(_sensor_config);
        _pimpl->ImageManagerInit();
        _pimpl->Start();
    });
}

void NvCamera::DeInit() {
    std::call_once(flag_deinit, [&](){
        _pimpl->Deinit();
        _pimpl->ImageManagerDeInit();
    });
}

int32_t NvCamera::RegisterProcess(FrameAvailableCallback callback, uint32_t sensor) {
    return _pimpl->RegisterProcess(callback, sensor);
}

int32_t NvCamera::GetImageData(std::string& image_data, uint32_t sensor) {
    return _pimpl->GetImageData(image_data, sensor);
}

double NvCamera::GetImageTimeStamp(uint32_t sensor) {
    return _pimpl->GetImageTimeStamp(sensor);
}

uint32_t NvCamera::GetImageHeight(uint32_t sensor) {
    return _pimpl->GetImageHeight(sensor);
}

uint32_t NvCamera::GetImageWidth(uint32_t sensor) {
    return _pimpl->GetImageWidth(sensor);
}

uint32_t NvCamera::GetFrameID(uint32_t sensor) {
    return _pimpl->GetFrameID(sensor);
}

}
}
}