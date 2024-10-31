#include "interface.h"
#include "camera_app.h"

namespace app {

CameraApp camera_app_;

void registe(const std::string &config_path, cuda_data_callback callback) {
    camera_app_.start(config_path, callback);
}

}  // namespace app
