#pragma once
#include <cmath>
#include <fstream>  // std::ifstream
#include <functional>
#include <iostream>  // std::cout
#include <memory>
#include <vector>
#include <google/protobuf/text_format.h>
#include <rosbag2_cpp/reader.hpp>
#include "bag_message.hpp"
#include "google/protobuf/descriptor.h"
#include "rosbag2_storage/serialized_bag_message.hpp"
#include "proto/localization/localization.pb.h"
#include "proto/soc/point_cloud.pb.h"
#include "proto/soc/raw_point_cloud.pb.h"

// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace bag {

#define ANGLE_TABLE_BITS_ALL 0x03ff

#define MAX_AZI_LEN (36000 * 256)
#define M_PI 3.14159265358979323846 /* pi */

#pragma pack(push, 1)

struct DataHeader {
    uint8_t laser_num;
    uint8_t block_num;
    uint8_t first_block_return;
    uint8_t dis_unit;
    uint8_t return_num;
    uint8_t flags;
};

struct ChannelData {
    uint16_t distance;
    uint8_t reflectivity;
    uint8_t confidence;
};

struct DataBlock {
    uint16_t azimuth;
    uint8_t fine_azimuth;
    ChannelData channel_data[128];
};

struct DataBody {
    DataBlock data_block[2];
    uint32_t crc_32;
};

struct DataTail {
    uint8_t reserver1[6];
    uint8_t high_temperature_flag;
    uint8_t reserver2[11];
    int16_t motor_speed;
    uint32_t timestamp;
    uint8_t return_mode;
    uint8_t factory_info;
    uint8_t data_time[6];
    uint32_t sequence;
    uint32_t crc_32;
};

struct PointCloudFrameDEA {
    uint8_t packet_header[6];
    DataHeader data_header;
    DataBody data_body;
    DataTail data_tail;
    uint8_t internet_safe[32];
};

struct SubPointCloudFrameDEA {
    DataHeader data_header;
    DataBody data_body;
    DataTail data_tail;
};

#pragma pack()

struct AngleTable {
    uint32_t set_bits = 0;
    double vangle[96] = {0};
    double hangle[1500] = {0};
    double vangle_offset[4][1500] = {{0}};
    double hangle_offset[4][96] = {{0}};
};

struct PandarATCorrectionsHeader {
    uint8_t delimiter[2];
    uint8_t version[2];
    uint8_t channel_number;
    uint8_t mirror_number;
    uint8_t frame_number;
    uint8_t frame_config[8];
    uint8_t resolution;
};

struct PointXYZ {
    double x;
    double y;
    double z;
    uint8_t reflectivity;
    uint8_t confidence;
};

struct PointCloud {
    PointXYZ point[128];
    float vertical_angle[128];
    float code_wheel_angle;
};

struct PandarATCorrections {
    PandarATCorrectionsHeader header;
    uint32_t start_frame[3];
    uint32_t end_frame[3];
    int32_t azimuth[128];
    int32_t elevation[128];
    int8_t azimuth_offset[23040];
    int8_t elevation_offset[23040];
    uint8_t SHA256[32];

    std::array<float, MAX_AZI_LEN> sin_map;  //#define MAX_AZI_LEN (36000 * 256)
    std::array<float, MAX_AZI_LEN> cos_map;

    PandarATCorrections() {
        for (int i = 0; i < MAX_AZI_LEN; ++i) {
            sin_map[i] = std::sin(2 * i * M_PI / MAX_AZI_LEN);
            cos_map[i] = std::cos(2 * i * M_PI / MAX_AZI_LEN);
        }
    }

    static const int STEP3 = 200 * 256;

    int8_t getAzimuthAdjustV3(uint8_t ch, uint32_t azi) const {
        unsigned int i = std::floor(1.f * azi / STEP3);
        unsigned int l = azi - i * STEP3;
        float k = 1.f * l / STEP3;
        return round((1 - k) * azimuth_offset[ch * 180 + i] + k * azimuth_offset[ch * 180 + i + 1]);
    }

    int8_t getElevationAdjustV3(uint8_t ch, uint32_t azi) const {  //获取垂直角修正量
        unsigned int i = std::floor(1.f * azi / STEP3);            //计算azi / STEP3 的整数部分   2°一步
        unsigned int l = azi - i * STEP3;                          //计算azi / STEP3 的余数部分
        float k = 1.f * l / STEP3;
        return round((1 - k) * elevation_offset[ch * 180 + i] + k * elevation_offset[ch * 180 + i + 1]);
    }
};

struct PointField {
    double x;
    double y;
    double z;
    uint32_t time;
    double distance;
    double pitch;
    double yaw;
    uint32_t intensity;
    uint32_t ring;
    uint32_t block;
};

class MessageProcess {
   public:
    MessageProcess(const std::string& config_path);
    MessageProcess(const std::string& uri, const std::string& storage_id);
    ~MessageProcess();
    static MessageProcess& Instance(const std::string& config_path);
    static MessageProcess& Instance(const std::string& uri, const std::string& storage_id);
    int Process(const std::shared_ptr<rosbag2_storage::SerializedBagMessage> inMessage, BagMessage& outMessage, const std::string topicName);
    int Parse(const hozon::soc::RawPointCloud& rawPointCloud, hozon::soc::PointCloud& pointCloud);

   private:
    PandarATCorrections m_PandarAT_corrections;
    bool if_load_correction_ = false;
    bool if_load_extrinsics_ = false;
    std::uint32_t frame_id = 0;
    static MessageProcess* instance_;
    std::shared_ptr<hozon::localization::Localization> start_dr;
    std::shared_ptr<hozon::localization::Localization> end_dr;
    Eigen::Affine3f lidar_extrinsics = Eigen::Affine3f::Identity();  //外参
    Eigen::Affine3f lidar_start_pose = Eigen::Affine3f::Identity();  //起始位姿
    Eigen::Affine3f lidar_end_pose = Eigen::Affine3f::Identity();    //终止位姿
    std::string lidar_correction_path;
    std::string lidar_extrinsics_path;
    std::unique_ptr<rosbag2_cpp::Reader> _reader = nullptr;

    // testcode:
    // uint64_t max_point_timestamp = 0;

    void init();
    template <typename ProtoType>
    void SerializedBagMessage2Proto(const std::shared_ptr<rosbag2_storage::SerializedBagMessage>& inMessage, ProtoType& protoMsg);
    void Proto2BagMessage(const google::protobuf::Message& proto_msg, BagMessage& outMessage);
    int loadCorrectionString(const char* data);
    bool loadCorrectionFile(const std::string& correction_path);
    bool LoadExtrinsics(const std::string& extrinsics_path);
    void motion_compensate(hozon::soc::PointCloud& pointCloud, const Eigen::Affine3f& start_pose, const Eigen::Affine3f& end_pose, const Eigen::Affine3f& lidar_extrinsics, uint32_t pack_num);
    Eigen::Affine3f location_parse(const uint8_t* cm_buf, const uint32_t len);
    Eigen::Affine3f DrToAffine3f(std::shared_ptr<hozon::localization::Localization> proto);
    bool read_Attachment(const std::string file_name, std::string& data);
    void LoadLidarQuaternion(YAML::Node& ext_file);
    std::shared_ptr<YAML::Node> lidarExtrinsicsParaConvert(std::string& json);
    std::string RePrecision(double num);
};
}  // namespace bag
}  // namespace netaos
}  // namespace hozon
