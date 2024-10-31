#ifndef LIDAR_POINT_CLOUD_PARSER_H
#define LIDAR_POINT_CLOUD_PARSER_H

#include <memory>
#include <mutex>
#include <vector>
#include <chrono>
#include <ctime>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "yaml-cpp/yaml.h"

#include "common/logger.h"
#include "protocol/point_cloud.h"
#include "publish/point_cloud_pub.h"
#include "proto/soc/point_cloud.pb.h"
#include "faultmessage/lidar_fault_report.h"

#include "cm/include/proto_cm_reader.h"
#include "proto/dead_reckoning/dr.pb.h"
#include "subdr/sub_dead_reckoning.h"
#include "cfg/include/config_param.h"
#include "faultmessage/lidar_status_report.h"



#define MAX_AZI_LEN (36000 * 256)

namespace hozon {
namespace ethstack {
namespace lidar {

#define ANGLE_TABLE_BITS_ALL 0x03ff

#define MAX_AZI_LEN (36000 * 256)
#define M_PI 3.14159265358979323846 /* pi */
#define CONFIG_FILE_PATH "/app/runtime_service/neta_lidar/conf/neta_lidar.yaml"


#pragma pack(push, 1)

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

    std::array<float, MAX_AZI_LEN> sin_map;         //#define MAX_AZI_LEN (36000 * 256)
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
        return round((1 - k) * azimuth_offset[ch * 180 + i] +
                    k * azimuth_offset[ch * 180 + i + 1]);
    }
    int8_t getElevationAdjustV3(uint8_t ch, uint32_t azi) const {           //获取垂直角修正量
        unsigned int i = std::floor(1.f * azi / STEP3);                       //计算azi / STEP3 的整数部分   2°一步
        unsigned int l = azi - i * STEP3;                                     //计算azi / STEP3 的余数部分
        float k = 1.f * l / STEP3;
        return round((1 - k) * elevation_offset[ch * 180 + i] +
                    k * elevation_offset[ch * 180 + i + 1]);
    }
    
};

#pragma pack(pop)

// using  hozon::ethstack::lidar::ProtoDRReader;

class PointCloudParser {
   public:
    static PointCloudParser& Instance();
    virtual ~PointCloudParser();

    void Parse(uint8_t* dataptr, uint32_t size);

    bool isPointCloudAddReady();
    void SetLidarFrameName(std::string name);
    void SetLidarSerialNumber(std::string name);
    int loadCorrectionString(char* data);
    bool loadCorrectionFile();
    void processProto();
    int m_iGetCorrectionCount;
    std::string m_sLidarCorrectionFile;
    PandarATCorrections m_PandarAT_corrections;
    // Eigen::Affine3f DrToAffine3f(std::shared_ptr<hozon::dead_reckoning::DeadReckoning>);
    Eigen::Affine3f DrToAffine3f(std::shared_ptr<hozon::localization::Localization>);

    void motion_compensate(const std::shared_ptr<hozon::ethstack::lidar::PointCloudFrame> send_data_ptr_,
        const Eigen::Affine3f& start_pose, const Eigen::Affine3f& end_pose,
        const Eigen::Affine3f& lidar_extrinsics,int block_num);

    bool LoadExtrinsics(Eigen::Affine3f& ext);
    void SaveNoMotionCompensate(const std::shared_ptr<hozon::ethstack::lidar::PointCloudFrame> send_data_ptr,const Eigen::Affine3f& lidar_extrinsics);
    void Init();

   private:
    PointCloudParser();
    std::shared_ptr<hozon::ethstack::lidar::PointCloudFrame> send_data_ptr_;
    std::shared_ptr<AngleTable> angle_table_ptr_;
    std::mutex angle_table_mutex_;
    std::string lidar_frame_name_;
    std::string lidar_serial_number_;
    std::vector<PointField> pointfieldvec;

    hozon::ethstack::lidar::ProtoDRReader reader;
    std::shared_ptr<hozon::localization::Localization> start_dr;
    std::shared_ptr<hozon::localization::Localization> end_dr;
    
    Eigen::Affine3f lidar_extrinsics = Eigen::Affine3f::Identity();           //外参
    Eigen::Affine3f lidar_start_pose = Eigen::Affine3f::Identity();           //起始位姿
    Eigen::Affine3f lidar_end_pose = Eigen::Affine3f::Identity();             //终止位姿

};

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // LIDAR_POINT_CLOUD_PARSER_H