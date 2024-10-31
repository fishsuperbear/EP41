#include "camera_device_callback.h"

#include "camera_app.h"

CameraDeviceCallback::~CameraDeviceCallback() {
}

CAMERA_DEVICE_OPENTYPE CameraDeviceCallback::RegisterOpenType() {
    return open_type_;
}

CAMERA_DEVICE_OPENMODE CameraDeviceCallback::RegisterOpenMode() {
    return CAMERA_DEVICE_OPENMODE_SUB;
}

bool CameraDeviceCallback::init(const std::shared_ptr<ICameraDevice> &device, const std::shared_ptr<CameraConfig> &config) {
    pcameradevice_ = device;
    pconfig_ = config;

    return true;
}

s32 CameraDeviceCallback::RegisterCallback() {
    int res = 0;

    CameraDeviceGpuDataCbRegInfo gpudatareginfo;
    for (int i = 0; i < pconfig_->mutable_config()->size(); i++) {
        gpudatareginfo.opentype = getOpenType(pconfig_->mutable_config(i)->open_type());
        gpudatareginfo.blocktype = (CAMERA_DEVICE_BLOCK_TYPE)pconfig_->mutable_config(i)->block_type();
        gpudatareginfo.sensortype = (CAMERA_DEVICE_SENSOR_TYPE)pconfig_->mutable_config(i)->sensor_type();
        gpudatareginfo.sensorindex = pconfig_->mutable_config(i)->sensor_index();
        gpudatareginfo.gpuimgtype = (CAMERA_DEVICE_GPUDATACB_IMGTYPE)pconfig_->mutable_config(i)->gpu_image_type();
        gpudatareginfo.interpolation = (CAMERA_DEVICE_GPUDATACB_INTERPOLATION)pconfig_->mutable_config(i)->inter_polation();
        gpudatareginfo.busecaptureresolution = pconfig_->mutable_config(i)->buse_capture_resolution();
        gpudatareginfo.customwidth = pconfig_->mutable_config(i)->custom_width();
        gpudatareginfo.customheight = pconfig_->mutable_config(i)->custom_height();
        gpudatareginfo.busecaptureframerate = pconfig_->mutable_config(i)->buse_capture_framerate();
        gpudatareginfo.rotatedegrees = pconfig_->mutable_config(i)->rotate_degrees();
        gpudatareginfo.pcontext = nullptr;
        res = pcameradevice_->RegisterGpuDataCallback(&gpudatareginfo, CameraApp::handle_cuda_buffer);
        if (res < 0) {
            printf("RegisterDataCallback failed! ret: %d\n", res);
        }
    }

    return res;
}

CAMERA_DEVICE_OPENTYPE CameraDeviceCallback::getOpenType(OpenType open_type) {
    switch (open_type) {
        case OpenType::OPENTYPE_MULTIROUP_SENSOR_DESAY:
            // SUB not open to user, camera_app only has consumer interface, user don't need to known sub or not-sub.
            return CAMERA_DEVICE_OPENTYPE::CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY_SUB;
        default:
            return (CAMERA_DEVICE_OPENTYPE)open_type;
    }
}
