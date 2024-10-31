#ifndef _CAMERA_APP_H_
#define _CAMERA_APP_H_

#include "interface.h"
#include "camera_device_callback.h"
#include "framework/cyber.h"
#include "camera_config.pb.h"

class CameraApp {
public:
    CameraApp();
    ~CameraApp();

    void start(const std::string &config_path, app::cuda_data_callback callback);

    static void handle_cuda_buffer(struct CameraDeviceGpuDataCbInfo* i_pbufferinfo);
    static void handle_event(CameraDeviceEventCbInfo* i_peventcbinfo);

private:
    std::shared_ptr<CameraConfig> pconfig_;

    CameraDeviceCallback* pcallback_ = nullptr;
    std::shared_ptr<ICameraDevice> pcameradevice_ = nullptr;
    static app::cuda_data_callback cuda_data_callback_;
};

#endif  // _CAMERA_APP_H_
