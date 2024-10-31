#ifndef _CAMERA_INTERFACE_H_
#define _CAMERA_INTERFACE_H_

#include <iostream>

#include "camera_types.hpp"

namespace app {

typedef void(*cuda_data_callback)(CameraDeviceGpuDataCbInfo *data);

void registe(const std::string &config_path, cuda_data_callback callback);

}  // namespace app

#endif // _CAMERA_INTERFACE_H_
