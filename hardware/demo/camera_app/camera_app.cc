#include "camera_app.h"

app::cuda_data_callback CameraApp::cuda_data_callback_;

CameraApp::CameraApp() {
}

CameraApp::~CameraApp() {
}

void CameraApp::start(const std::string &config_path, app::cuda_data_callback callback) {
    cuda_data_callback_ = callback;

    pconfig_ = std::make_shared<CameraConfig>();
    if (!netaos::framework::common::GetProtoFromFile(config_path, pconfig_.get())) {
        printf("parse %s failed!\n", config_path.c_str());
        return;
    }
    
    int size = pconfig_->mutable_config()->size();
    printf("size of camera_config: %d\n", size);
    printf("========================\n");
    for (int i = 0; i < size; i++) {
        printf("no: %d\n", pconfig_->mutable_config(i)->no());
        printf("open_type: %s\n", OpenType_Name(pconfig_->mutable_config(i)->open_type()).c_str());
        printf("block_type: %s\n", BlockType_Name(pconfig_->mutable_config(i)->block_type()).c_str());
        printf("sensor_type: %s\n", SensorType_Name(pconfig_->mutable_config(i)->sensor_type()).c_str());
        printf("sensor_index: %d\n", pconfig_->mutable_config(i)->sensor_index());
        printf("gpu_image_type: %s\n", GpuImageType_Name(pconfig_->mutable_config(i)->gpu_image_type()).c_str());
        printf("inter_polation: %s\n", InterPolation_Name(pconfig_->mutable_config(i)->inter_polation()).c_str());
        printf("buse_capture_resolution: %d\n", pconfig_->mutable_config(i)->buse_capture_resolution());
        printf("custom_width: %d\n", pconfig_->mutable_config(i)->custom_width());
        printf("custom_height: %d\n", pconfig_->mutable_config(i)->custom_height());
        printf("buse_capture_framerate: %d\n", pconfig_->mutable_config(i)->buse_capture_framerate());
        printf("rotate_degrees: %d\n", pconfig_->mutable_config(i)->rotate_degrees());
        printf("========================\n");
    }

    pcallback_ = new CameraDeviceCallback();
    ICameraDeviceSession *psession;
    pcameradevice_ = ICameraDevice::GetInstance(HAL_CAMERA_VERSION_0_1, false);
    pcallback_->init(pcameradevice_, pconfig_);
    s32 res = pcameradevice_->CreateCameraSession(pcallback_, &psession);
    if (res < 0) {
        printf("Camera Device Open failed! res: %d\n", res);
        return;
    }
}

void CameraApp::handle_cuda_buffer(struct CameraDeviceGpuDataCbInfo* i_pbufferinfo) {
    printf("handle_cuda_buffer+++++++++++++++++++++++\n");
    if (cuda_data_callback_) {
        cuda_data_callback_(i_pbufferinfo);
    }
}

void CameraApp::handle_event(CameraDeviceEventCbInfo* i_peventcbinfo)
{
	// currently do nothing
}
