#include <iostream>
#include <fstream>

#include "nvs_sensor_executor.h"
#include "adf-lite/include/ds/builtin_types.h"
#include "adf/include/node_proto_register.h"
#include "adf-lite/service/rpc/lite_rpc.h"
#include "adf-lite/include/writer.h"
#include "proto/soc/sensor_image.pb.h"


uint16_t vMasks = 4091;
std::string config_name = "conf/nvs_top_config.yaml";
bool file_dump = false;

NvsSensorExecutor::NvsSensorExecutor() {}

NvsSensorExecutor::~NvsSensorExecutor() {}

int32_t NvsSensorExecutor::AlgInit() {
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_0", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_1", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_3", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_4", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_5", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_6", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_7", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_8", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_9", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_10", hozon::soc::Image)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/camera_11", hozon::soc::Image)

    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_0", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_1", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_3", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_4", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_5", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_6", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_7", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_8", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_9", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_10", hozon::soc::CompressedImage)
    REGISTER_PROTO_MESSAGE_TYPE("/soc/encoded_camera_11", hozon::soc::CompressedImage)

    NODE_LOG_INFO << "Init NvsSensorExecutor.";
    RegistAlgProcessFunc("nvs_cam", std::bind(&NvsSensorExecutor::NvsCamProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("h265_cam", std::bind(&NvsSensorExecutor::EncodeImageProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("replay_cam", std::bind(&NvsSensorExecutor::ProtoImageProcess, this, std::placeholders::_1));

    return 0;
}

void NvsSensorExecutor::AlgRelease() {
}

void NvsSensorExecutor::WriteFile(const std::string& name, uint8_t* data, uint32_t size) {
    std::ofstream of(name);

    if (!of) {
        NODE_LOG_ERROR << "Fail to open " << name;
        return;
    }

    of.write((const char*)data, size);
    of.close();
    NODE_LOG_INFO << "Succ to write " << name;

    return;
}

void NvsSensorExecutor::DumpCudaImage(const std::string& name, std::shared_ptr<NvsImageCUDA> packet) {
    uint8_t* local_ptr = (uint8_t*)malloc(packet->size);

    /* Instruct CUDA to copy the packet data buffer to the target buffer */
    uint32_t cuda_rt_err = cudaMemcpy(local_ptr,
                                packet->cuda_dev_ptr,
                                packet->size,
                                cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuda_rt_err) {
        NODE_LOG_ERROR << "Failed to issue copy command, ret " << packet->cuda_dev_ptr;
        return;
    }

    std::string dump_name = name + "_" + std::to_string(packet->data_time_sec);
    WriteFile(dump_name, local_ptr, packet->size);

    free(local_ptr);
}

template <typename T>
std::shared_ptr<T> NvsSensorExecutor::GetImageData(Bundle* input, const std::string& name) {
    std::shared_ptr<T> camera_ptr = std::static_pointer_cast<T>(input->GetOne(name));
    if (camera_ptr == nullptr) {
        NODE_LOG_ERROR << "Fail to recv " << name;
        return nullptr;
    }

    freq_checker.say(name);
    return camera_ptr;
}

int32_t NvsSensorExecutor::NvsCamProcess(Bundle* input) {
    for (uint8_t i = 0 ; i < 12; i++) {
        uint16_t mask = (0x1 << i);
        if ((vMasks & mask) == mask) {
            std::string topic = "camera_" + std::to_string(i);
            std::shared_ptr<NvsImageCUDA> data_ptr = GetImageData<NvsImageCUDA>(input, topic);
            if (data_ptr != nullptr) {
                NODE_LOG_DEBUG << "nvs image : " << topic << SET_PRECISION(10) << FIXED
                        << " publish_stamp : " << data_ptr->__header.timestamp_real_us
                        << " data_time_sec : " << data_ptr->data_time_sec;
                if (file_dump == true) {
                    DumpCudaImage("img_"+ std::to_string(i), 
                            std::static_pointer_cast<NvsImageCUDA>(data_ptr));
                }
            }
        }
    }
    return 0;
}

int32_t NvsSensorExecutor::EncodeImageProcess(Bundle* input) {
    for (uint8_t i = 0 ; i < 12; i++) {
        uint16_t mask = (0x1 << i);
        if ((vMasks & mask) == mask) {
            std::string topic = "/soc/encoded_camera_" + std::to_string(i);
            BaseDataTypePtr data_ptr = GetImageData<BaseData>(input, topic);
            std::shared_ptr<hozon::soc::CompressedImage> image = 
                    std::static_pointer_cast<hozon::soc::CompressedImage>(data_ptr->proto_msg);
        }
    }
    return 0;
}

int32_t NvsSensorExecutor::ProtoImageProcess(Bundle* input) {
    for (uint8_t i = 0 ; i < 12; i++) {
        uint16_t mask = (0x1 << i);
        if ((vMasks & mask) == mask) {
            std::string topic = "/soc/camera_" + std::to_string(i);
            std::shared_ptr<NvsImageCUDA> data_ptr = GetImageData<NvsImageCUDA>(input, topic);
            if (data_ptr != nullptr) {
                NODE_LOG_DEBUG << "proto image : " << topic << SET_PRECISION(10) << FIXED
                        << " publish_stamp : " << data_ptr->__header.timestamp_real_us
                        << " data_time_sec : " << data_ptr->data_time_sec;
                if (file_dump == true) {
                    DumpCudaImage("img_"+ std::to_string(i), 
                            std::static_pointer_cast<NvsImageCUDA>(data_ptr));
                }
            }
        }
    }
    return 0;
}