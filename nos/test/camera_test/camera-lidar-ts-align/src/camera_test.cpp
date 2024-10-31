#include <iostream>
#include <memory>
#include <map>
#include <unistd.h>
#include <fstream>
#include <getopt.h>
#include "adf/include/log.h"
#include "proto/test/soc/for_test.pb.h"
#include "adf/include/node_base.h"
#include "proto/soc/sensor_image.pb.h"
#include "proto/soc/point_cloud.pb.h"
#include "proto/soc/sensor_imu_ins.pb.h"
#include "proto/soc/chassis.pb.h"
#include "adf/include/data_types/image/orin_image.h"

class FreqChecker {
    using checker_time = std::chrono::time_point<std::chrono::system_clock>;

   public:
    FreqChecker() = default;
    void say(const std::string& unique_name, uint64_t sample_cnt = 100);

   private:
    std::unordered_map<std::string, std::pair<uint64_t, checker_time>> freq_map_;
};

void FreqChecker::say(const std::string& unique_name, uint64_t sample_cnt) {
    if (freq_map_.find(unique_name) == freq_map_.end()) {
        freq_map_[unique_name] = std::make_pair(1, std::chrono::system_clock::now());
    } else {
        freq_map_[unique_name].first++;
    }

    if (freq_map_[unique_name].first == sample_cnt) {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = now - freq_map_[unique_name].second;
        NODE_LOG_INFO << "check " << unique_name << " frequency: " << sample_cnt / diff.count() << " Hz";
        freq_map_[unique_name].second = now;
        freq_map_[unique_name].first = 0;
    }
}


using namespace hozon::netaos::log;
using namespace hozon::netaos::adf;

uint16_t vMasks = 4091;
std::string config_name = "conf/nvs_cam_sample_conf2.yaml";
bool file_dump = false;

class CameraTest : public hozon::netaos::adf::NodeBase {
public:
    CameraTest() {}

    ~CameraTest() {}

    virtual int32_t AlgInit() {
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

        REGISTER_PROTO_MESSAGE_TYPE("pointcloud", hozon::soc::PointCloud)
        REGISTER_PROTO_MESSAGE_TYPE("imu_ins", hozon::soc::ImuIns)
        REGISTER_PROTO_MESSAGE_TYPE("chassis", hozon::soc::Chassis)

        RegistAlgProcessWithProfilerFunc("nvs_cam", std::bind(&CameraTest::NvsCamProcess, this, std::placeholders::_1, std::placeholders::_2));
        RegistAlgProcessWithProfilerFunc("h265_cam", std::bind(&CameraTest::EncodeImageProcess, this, std::placeholders::_1, std::placeholders::_2));
        RegistAlgProcessWithProfilerFunc("replay_cam", std::bind(&CameraTest::ProtoImageProcess, this, std::placeholders::_1, std::placeholders::_2));

        return 0;
    }

    int32_t AlgProcess(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
        BaseDataTypePtr base_ptr = input->GetOne("workresult");
        if (base_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv data.";
            return -1;
        }

        return 0;
    }

    void WriteFile(const std::string& name, uint8_t* data, uint32_t size) {
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

    void DumpCudaImage(const std::string& name, std::shared_ptr<NvsImageCUDA> packet) {
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

        NODE_LOG_INFO << "width : " << packet->width << " height : " << packet->height 
                << " size : "<< packet->size << " step : "<< packet->step;

        std::string dump_name = name + "_" + std::to_string(packet->data_time_sec);
        WriteFile(dump_name, local_ptr, packet->size);

        free(local_ptr);
    }

    template <typename T>
    std::shared_ptr<T> GetImageData(hozon::netaos::adf::NodeBundle* input, const std::string& name) {
        std::shared_ptr<T> camera_ptr = std::static_pointer_cast<T>(input->GetOne(name));
        if (camera_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv " << name;
            return nullptr;
        }

        checker.say(name);
        return camera_ptr;
    }

    int32_t NvsCamProcess(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
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

        std::shared_ptr<hozon::soc::PointCloud> pointcloud_ptr = 
                std::static_pointer_cast<hozon::soc::PointCloud>(input->GetOne("pointcloud")->proto_msg);
        if (pointcloud_ptr == nullptr) {
            NODE_LOG_ERROR << "Fail to recv " << "pointcloud";
        }

        return 0;
    }

    int32_t EncodeImageProcess(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
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

    int32_t ProtoImageProcess(hozon::netaos::adf::NodeBundle* input,
                            const hozon::netaos::adf::ProfileToken& token) {
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

        BaseDataTypePtr base_point_ptr = input->GetOne("pointcloud");
        if (base_point_ptr != nullptr) {
            std::shared_ptr<hozon::soc::PointCloud> pointcloud_ptr = 
                std::static_pointer_cast<hozon::soc::PointCloud>(base_point_ptr->proto_msg);
            NODE_LOG_DEBUG << "lidar publish_stamp : " << SET_PRECISION(10) << hozon::netaos::log::FIXED << pointcloud_ptr->header().publish_stamp() 
                << " sensor_stamp : " << pointcloud_ptr->header().sensor_stamp().lidar_stamp();
        }

        BaseDataTypePtr base_chassis_ptr = input->GetOne("chassis");
        if (base_chassis_ptr != nullptr) {
            std::shared_ptr<hozon::soc::Chassis> chassis_ptr = 
                    std::static_pointer_cast<hozon::soc::Chassis>(base_chassis_ptr->proto_msg);
            NODE_LOG_DEBUG << "chassis publish_stamp : " << SET_PRECISION(10) << hozon::netaos::log::FIXED << chassis_ptr->header().publish_stamp() 
                << " sensor_stamp : " << chassis_ptr->header().sensor_stamp().chassis_stamp();
        }

        BaseDataTypePtr base_imuins_ptr = input->GetOne("imu_ins");
        if (base_imuins_ptr != nullptr) {
            std::shared_ptr<hozon::soc::ImuIns> imu_ins_ptr = 
                std::static_pointer_cast<hozon::soc::ImuIns>(base_imuins_ptr->proto_msg);
            NODE_LOG_DEBUG << "ImuIns publish_stamp : " << SET_PRECISION(10) << hozon::netaos::log::FIXED << imu_ins_ptr->header().publish_stamp() 
                << " sensor_stamp : " << imu_ins_ptr->header().sensor_stamp().imuins_stamp();
        }

        return 0;
    }

    virtual void AlgRelease() { 
    }

private:
    FreqChecker checker;
};

void Parse(int argc, char* argv[]) {
    struct option long_opts[] = {
        {"help",         no_argument,         nullptr,    'h'},
        {"config-file",  required_argument,   nullptr,    'c'},
        {"link-num",     required_argument,   nullptr,    'l'},
        {"dump",         no_argument,         nullptr,    'd'},
        {0, 0, 0, 0}
    };

    int opt_index = 0;
    while (1) {
        int c = getopt_long(argc, argv, "hc:l:d", long_opts, &opt_index);
        if (c == -1) {
            break;
        }

        switch (c) {
        case 'c':
            config_name = std::string(optarg);
            break;
        case 'l':
            {
                char *token = std::strtok(optarg, " ");
                uint8_t i = 0;
                vMasks = 0;
                while (token != NULL) {
                    uint16_t mask = std::stoi(token, nullptr, 2);
                    vMasks |= mask << (4 * i);
                    token = std::strtok(NULL, " ");
                    i++;
                }
            }
            break;
        case 'd':
            file_dump = true;
            break;
        case 'h':
        default:
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    CameraTest recv_node;

    Parse(argc, argv);
    recv_node.InitLoggerStandAlone(config_name);
    recv_node.Start(config_name, true);
    recv_node.NeedStopBlocking();
    recv_node.Stop();

    return 0;
}