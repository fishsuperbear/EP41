#include <unistd.h>
#include "interface.h"

void data_cuda_callback(struct CameraDeviceGpuDataCbInfo* i_pbufferinfo) {
    printf("data_cuda_callback+++++++++++++++++++++++\n");
}

int main(int argc, char **argv) {

    std::string path = "/home/orin/szf/code/hardware/demo/camera_app/conf/camera_front_728.pb.txt";
    if (argc > 1) {
        path = argv[1];
    }

    app::registe(path, data_cuda_callback);

    while (1) {
        sleep(1);
    }

    return 0;
}
