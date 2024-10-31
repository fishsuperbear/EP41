#ifndef _CAMERA_DEVICE_CALLBACK_H_
#define _CAMERA_DEVICE_CALLBACK_H_

#include <functional>
#include "hal_camera.hpp"
#include "camera_config.pb.h"

class CameraDeviceCallback : public ICameraDeviceCallback {
public:
    ~CameraDeviceCallback();
    
    CAMERA_DEVICE_OPENTYPE RegisterOpenType() override;
	CAMERA_DEVICE_OPENMODE RegisterOpenMode() override;
    s32 RegisterCallback() override;

    bool init(const std::shared_ptr<ICameraDevice> &device, const std::shared_ptr<CameraConfig> &config);

private:
    CAMERA_DEVICE_OPENTYPE getOpenType(OpenType open_type);

    CAMERA_DEVICE_OPENTYPE open_type_ = CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY_SUB;
    std::shared_ptr<CameraConfig> pconfig_;
    std::shared_ptr<ICameraDevice> pcameradevice_ = nullptr;
};

#endif  // _CAMERA_DEVICE_CALLBACK_H_
