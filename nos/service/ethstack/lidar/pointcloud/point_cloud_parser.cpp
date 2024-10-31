#include "point_cloud_parser.h"
#include "common/logger.h"
#include "faultmessage/lidar_fault_report.h"
#include "protocol/point_cloud.h"
#include <cmath>
#include <arpa/inet.h>
#include <sys/time.h>
#include <time.h>
#include <time.h>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <json/json.h>

namespace hozon {
namespace ethstack {
namespace lidar {

const int LIDAR_POINTCLOUD_HEIGHT = 96;
const int LIDAR_POINTCLOUD_WIDTH = 480;
const uint16_t LIDAR_POINTCLOUD_BLOCK_START = 0xFFEE;

const double LIDAR_ANGLE_RESOLT = 1.0L/256.0L;
const double LIDAR_DISTANCE_RESOLT = 1.0L/256.0L;

const double LIDAR_ANGLE_OFFSET_RESOLT = 0.001L;

const double LIDAR_PI = ::acos(-1.0L);
const double LIDAR_1D_ARC = LIDAR_PI / 180.0L;
// const uint32_t LIDAR_1SEC_NSEC = 1000000000;
#define d2arc(x) (x * LIDAR_1D_ARC)

const uint8_t LIDAR_PACKAGE_FRAME_HEAD = 0;
const uint8_t LIDAR_PACKAGE_FRAME_TAIL = 2;

const int32_t LIDAR_VANGLE_TABLE_SIZE = 96 * 2;
const int32_t LIDAR_HANGLE_TABLE_SIZE = 1500 * 2;
const int32_t LIDAR_VANGLE_OFFSET_TABLE_NUM = 1500;
const int32_t LIDAR_VANGLE_OFFSET_TABLE_SIZE = LIDAR_VANGLE_OFFSET_TABLE_NUM * 2;
const int32_t LIDAR_HANGLE_OFFSET_TABLE_NUM = 96;
const int32_t LIDAR_HANGLE_OFFSET_TABLE_SIZE = LIDAR_HANGLE_OFFSET_TABLE_NUM * 2;
const int32_t LIDAR_ANGLE_OFFSET_TABLE_INDEX_NUM = 4;
const int MAX_LOAD_SIZE = 1024;
const int target_length = 47176;

static bool initialized = false;
static bool framing_flag = false;
static bool start_frame_flag = false;
static bool print_flag = false;
static bool loadCorrectionFileFlag = false;
int count_point = 0;
int pack_num = 0;
int32_t Azimuth_last;
float fov_angle = 0;
int frame_id = 0;     //点云帧编号
int frameid;          //hesai帧编号  
uint32_t seq;
uint32_t seq_last;
double start_time;
double end_time;
uint8_t motion_compensate_flag;
uint8_t save_no_motion_compensate_pcd_flag;

static std::mutex inst_mutex_;

PointCloudParser& PointCloudParser::Instance() {
  static PointCloudParser instance;
  return instance;
}

PointCloudParser::PointCloudParser()
        : send_data_ptr_(nullptr)
        , angle_table_ptr_(nullptr)
        , lidar_frame_name_("")
        , lidar_serial_number_("")
{
    
}

void PointCloudParser::Init() {
    int32_t ret = reader.Init(0,"/perception/parking/slam_location");
    if (ret < 0){
        LIDAR_LOG_ERROR << "DRReader Init fail.";
    }

    bool ans = LoadExtrinsics(lidar_extrinsics);
	  if (!ans){
      LIDAR_LOG_ERROR << "LoadExtrinsics load fail.";
    }

    YAML::Node config;
    if (0 == access(CONFIG_FILE_PATH, F_OK)) {
        LIDAR_LOG_INFO << "success get neta_lidar.yaml.";
        config = YAML::LoadFile(CONFIG_FILE_PATH);
        motion_compensate_flag = config["motion_compensate_flag"].as<uint8_t>();
        save_no_motion_compensate_pcd_flag = config["save_no_motion_compensate_pcd_flag"].as<uint8_t>();
    }
    else {
        LIDAR_LOG_ERROR << "fail to get neta_lidar.yaml.";
        LidarFaultReport::Instance().ReportAbstractFault(AbstractFaultObject::CONFIG_LOAD_ERROR, CURRENT_FAULT);
    }

    uint8_t lidar_history_status = hozon::ethstack::lidar::LidarStatusReport::Instance().GetLidarStatus();
    uint8_t lidar_current_status = 2;
    if (lidar_history_status != lidar_current_status){
        hozon::netaos::cfg::ConfigParam::Instance()->SetParam("system/lidar_status", lidar_current_status);
        LIDAR_LOG_TRACE << "success write lidar work status to cfg.";
        hozon::ethstack::lidar::LidarStatusReport::Instance().SetLidarStatus(lidar_current_status);
        LIDAR_LOG_TRACE << "success write lidar work status to LidarStatus class.";
    }

    loadCorrectionFileFlag = loadCorrectionFile();
    if(!loadCorrectionFileFlag){
      LidarFaultReport::Instance().ReportAbstractFault(AbstractFaultObject::CONFIG_LOAD_ERROR, CURRENT_FAULT);
      LIDAR_LOG_ERROR << "fail to load Correction File.";
    }
    else{
      LIDAR_LOG_INFO << "success to load Correction File.";
    }
}

PointCloudParser::~PointCloudParser() {
  reader.Deinit();
}

bool PointCloudParser::isPointCloudAddReady() {
    std::lock_guard<std::mutex> lock(angle_table_mutex_);
    if (nullptr == angle_table_ptr_) {
        return false;
    }
    return (ANGLE_TABLE_BITS_ALL == angle_table_ptr_->set_bits);
}

void PointCloudParser::SetLidarFrameName(std::string name) {
    lidar_frame_name_ = name;
}

void PointCloudParser::SetLidarSerialNumber(std::string name) {
    lidar_serial_number_ = name;
}

//单位为1/25600度
void PointCloudParser::Parse(uint8_t* dataptr, uint32_t size) {
    double soc_time = GetRealTimestamp();
    PointCloudFrameDEA* frame = reinterpret_cast<PointCloudFrameDEA*>(dataptr);         //把收到的数据包转为PointCloudFrameDEA类
    //加载校正文件
    if(!loadCorrectionFileFlag){
        bool ret = loadCorrectionFile();
        if(ret){
          loadCorrectionFileFlag = true;
        }
        return;
    }

    if (!send_data_ptr_) {
        send_data_ptr_ = PointCloudPub::Instance().GetSendDataPtr();
        send_data_ptr_->data.reserve(165000);
    }

    uint8_t laser_num = frame->data_header.laser_num;
    uint8_t block_num = frame->data_header.block_num;

    //判断头帧
    if( start_frame_flag == true  ){
        // struct timespec gnss_time;
        send_data_ptr_->header.seq = frame_id;
        send_data_ptr_->isBigEndian = 0;
        send_data_ptr_->width = 1;
        send_data_ptr_->pointStep = sizeof(PointField);
        send_data_ptr_->isDense = 1;

        std::tm timeinfo = {};
        timeinfo.tm_year = frame->data_tail.data_time[0];       // 年份从1900开始
        timeinfo.tm_mon = frame->data_tail.data_time[1] - 1;    // 月份从0开始
        timeinfo.tm_mday = frame->data_tail.data_time[2];
        timeinfo.tm_hour = frame->data_tail.data_time[3];
        timeinfo.tm_min = frame->data_tail.data_time[4];
        timeinfo.tm_sec = frame->data_tail.data_time[5];
        
        std::time_t time = std::mktime(&timeinfo);               // 使用mktime将tm结构体转换为time_t（自1970年1月1日以来的秒数）
        std::chrono::system_clock::time_point timepoint = std::chrono::system_clock::from_time_t(time);
        auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(timepoint);       // 转换为秒
        auto timestamp = seconds.time_since_epoch().count();
        send_data_ptr_->header.stamp.sec = timestamp;
        send_data_ptr_->header.stamp.nsec = (static_cast<uint64_t>(frame->data_tail.timestamp) * 1000);
        start_time = double(send_data_ptr_->header.stamp.sec) + double(send_data_ptr_->header.stamp.nsec) / LIDAR_1SEC_NSEC;
        LIDAR_LOG_TRACE << "sec is : "<<send_data_ptr_->header.stamp.sec 
                        <<"  nsec is : "<<send_data_ptr_->header.stamp.nsec;
        start_dr = reader.GetDeadReckoning();                     //获取第一列的dr数据
        lidar_start_pose = DrToAffine3f(start_dr);                //将proto数据转成Affine3f
        fov_angle = 0;
        pack_num = 0;
    }
    start_frame_flag = false;

    seq = frame->data_tail.sequence;
    //遍历接受到的UDP数据包
    for(int i = 0;i < block_num;i++){  
        //计算码盘角度
        uint16_t azimuth = frame->data_body.data_block[i].azimuth;
        uint8_t fine_azimuth = frame->data_body.data_block[i].fine_azimuth;
        int32_t Azimuth = azimuth * 256 + fine_azimuth;
        if(abs(Azimuth - Azimuth_last) > (50 * 25600))               
        {
            LIDAR_LOG_TRACE << "success Code_wheel_angle_last is: "<< Azimuth_last / 25600.f;
            LIDAR_LOG_TRACE << "success next frame Code_wheel_angle is: "<< Azimuth / 25600.f;
            framing_flag = true;
        }
        //获取Hesai的frameid
        int count = 0, field = 0;
        while (count < m_PandarAT_corrections.header.frame_number &&
               (((Azimuth + MAX_AZI_LEN -
                  m_PandarAT_corrections.start_frame[field]) %
                     MAX_AZI_LEN +
                 (m_PandarAT_corrections.end_frame[field] + MAX_AZI_LEN -
                  Azimuth) %
                     MAX_AZI_LEN) !=
                (m_PandarAT_corrections.end_frame[field] + MAX_AZI_LEN -
                 m_PandarAT_corrections.start_frame[field]) %
                    MAX_AZI_LEN)) {
          field = (field + 1) % m_PandarAT_corrections.header.frame_number;
          count++;
        }
        if (count >= m_PandarAT_corrections.header.frame_number) continue;
        frameid = field;

        // 遍历128线
        for(int j = 0;j<laser_num;j++){      
            auto azimuth = ((Azimuth + MAX_AZI_LEN - 
                            m_PandarAT_corrections.start_frame[field]) * 2 -          
                            m_PandarAT_corrections.azimuth[j] + 
                            m_PandarAT_corrections.getAzimuthAdjustV3(j, Azimuth) * 256);       //单位为：1/25600° 
            azimuth = (MAX_AZI_LEN + azimuth) % MAX_AZI_LEN;
            
            auto elevation = (m_PandarAT_corrections.elevation[j] + 
                              m_PandarAT_corrections.getElevationAdjustV3(j, Azimuth) * 256);     
            elevation = (MAX_AZI_LEN + elevation) % MAX_AZI_LEN;

            float distance = static_cast<float>(frame->data_body.data_block[i].channel_data[j].distance) * 0.004;      //单位是米
            if ((distance < 1.0) || (distance > 210.0)){         //根据测距范围去除噪点
              continue;
            }
            PointField point;
            point.distance = (double)distance;
            double xyDistance = distance * m_PandarAT_corrections.cos_map[(elevation)];
            point.x = xyDistance * m_PandarAT_corrections.sin_map[(azimuth)];
            point.y = xyDistance * m_PandarAT_corrections.cos_map[(azimuth)];
            point.z = point.distance * m_PandarAT_corrections.sin_map[(elevation)];
            point.time = frame->data_tail.timestamp;
            point.pitch = (double)elevation;
            point.yaw = (double)azimuth;
            // point.intensity = temp_intensity;
            point.intensity = (uint32_t)frame->data_body.data_block[i].channel_data[j].reflectivity;
            point.ring = (uint32_t)j;
            point.block = (uint32_t)pack_num;
            send_data_ptr_->data.push_back(point);
        } 
        pack_num++;
        fov_angle = fov_angle + 0.1;
        Azimuth_last = Azimuth;
    }
    if(seq - seq_last > 1){                            //判断丢包
      LIDAR_LOG_ERROR << "UDP package is lost,now seq is: "<<seq<<"  last seq is "<<seq_last<<", lost num is "<<(seq - seq_last);
    }
    seq_last = seq;

    //判断尾帧
    if( framing_flag == true ){

        send_data_ptr_->header.stamp.nsec = (static_cast<uint64_t>(frame->data_tail.timestamp) * 1000);
        std::tm timeinfo = {};
        char time_buf[128] = { 0 };
        snprintf(time_buf, sizeof(time_buf) - 1, "[%04d-%02d-%02d %02d:%02d:%02d.%09ld] ",
        timeinfo.tm_year = frame->data_tail.data_time[0],
        timeinfo.tm_mon = frame->data_tail.data_time[1] - 1,
        timeinfo.tm_mday = frame->data_tail.data_time[2],
        timeinfo.tm_hour = frame->data_tail.data_time[3],
        timeinfo.tm_min = frame->data_tail.data_time[4],
        timeinfo.tm_sec = frame->data_tail.data_time[5],
        send_data_ptr_->header.stamp.nsec);

        std::time_t time = std::mktime(&timeinfo);
        std::chrono::system_clock::time_point timepoint = std::chrono::system_clock::from_time_t(time);
        auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(timepoint);
        auto timestamp = seconds.time_since_epoch().count();
        send_data_ptr_->header.stamp.sec = timestamp;
        end_time = double(send_data_ptr_->header.stamp.sec) + double(send_data_ptr_->header.stamp.nsec) / LIDAR_1SEC_NSEC;
        LIDAR_LOG_TRACE << "lidar frame scan time is : "<< std::to_string((end_time - start_time) * 1000);
        // current timestamp in pointcloud header is the first frame timestamp,  normally maybe fill in the end frame timestamp
        if (send_data_ptr_->header.seq % 10 == 0) {
            LIDAR_LOG_INFO << "Send point cloud frame: "
                        << "frame_seq[" << send_data_ptr_->header.seq << "] "
                        << "point count[" << send_data_ptr_->data.size() <<"] "
                        << "stamp.sec[" << send_data_ptr_->header.stamp.sec << "] "
                        << "stamp.nsec[" << send_data_ptr_->header.stamp.nsec << "] "
                        << "time:" << time_buf
                        << "soc time is: "<< std::to_string(soc_time);
        }

        send_data_ptr_->height = send_data_ptr_->data.size();                                           //一帧点云的点数
        LIDAR_LOG_TRACE << "Parse send_data_ptr_->height is: "<<send_data_ptr_->height;
        send_data_ptr_->rowStep = send_data_ptr_->pointStep * send_data_ptr_->data.size();

        if(save_no_motion_compensate_pcd_flag == 1){
          LIDAR_LOG_INFO << "success save no motion compensate pcd ";    
          SaveNoMotionCompensate(send_data_ptr_,lidar_extrinsics);
        }
        
        end_dr = reader.GetDeadReckoning();                                                             //获取最后一列的dr数据    
        LIDAR_LOG_TRACE << " out reader.isGetData() is : "<<reader.isGetData();                                                     
        if((reader.isGetData() == 0) && (motion_compensate_flag == 1)){
          LIDAR_LOG_TRACE << " inside reader.isGetData() is : "<<reader.isGetData();
          lidar_end_pose = DrToAffine3f(end_dr);                                                          //将proto转成Affine3f
          motion_compensate(send_data_ptr_,lidar_start_pose,lidar_end_pose,lidar_extrinsics,pack_num);    //引入运动补偿
        }
        PointCloudPub::Instance().SetSendData(send_data_ptr_);
        LIDAR_LOG_INFO << "pointcloud frame size is: "<<sizeof(*send_data_ptr_.get());
        send_data_ptr_ = nullptr;
        start_frame_flag = true;
        frame_id++;
        LIDAR_LOG_INFO << "pointcloud frame pack_num is: "<<pack_num;
        LIDAR_LOG_TRACE << "end_frame ";
    }
    framing_flag = false;
}


int PointCloudParser::loadCorrectionString(char *data) {
  try {
    char *p = data;
    PandarATCorrectionsHeader header = *(PandarATCorrectionsHeader *)p;
    if (0xee == header.delimiter[0] && 0xff == header.delimiter[1]) {
      switch (header.version[1]) {
        case 3: {

        } break;
        case 5: {
          m_PandarAT_corrections.header = header;
          auto frame_num = m_PandarAT_corrections.header.frame_number;
          auto channel_num = m_PandarAT_corrections.header.channel_number;
          p += sizeof(PandarATCorrectionsHeader);
          memcpy((void *)&m_PandarAT_corrections.start_frame, p,
                 sizeof(uint32_t) * frame_num);
          p += sizeof(uint32_t) * frame_num;
          memcpy((void *)&m_PandarAT_corrections.end_frame, p,
                 sizeof(uint32_t) * frame_num);
          p += sizeof(uint32_t) * frame_num;
          memcpy((void *)&m_PandarAT_corrections.azimuth, p,
                 sizeof(int32_t) * channel_num);
          p += sizeof(int32_t) * channel_num;
          memcpy((void *)&m_PandarAT_corrections.elevation, p,
                 sizeof(int32_t) * channel_num);
          p += sizeof(int32_t) * channel_num;
          auto adjust_length = channel_num * 180;
          memcpy((void *)&m_PandarAT_corrections.azimuth_offset, p,
                 sizeof(int8_t) * adjust_length);
          p += sizeof(int8_t) * adjust_length;
          memcpy((void *)&m_PandarAT_corrections.elevation_offset, p,
                 sizeof(int8_t) * adjust_length);
          p += sizeof(int8_t) * adjust_length;
          memcpy((void *)&m_PandarAT_corrections.SHA256, p,
                 sizeof(uint8_t) * 32);
          p += sizeof(uint8_t) * 32;

          LIDAR_LOG_INFO << "frame_num: "<<frame_num;
          LIDAR_LOG_INFO <<"start_frame, end_frame:";
          for (int i = 0; i < frame_num; ++i) {
            m_PandarAT_corrections.start_frame[i] =
                m_PandarAT_corrections.start_frame[i] *
                m_PandarAT_corrections.header.resolution;           //resolution代表单位
            m_PandarAT_corrections.end_frame[i] =
                m_PandarAT_corrections.end_frame[i] *
                m_PandarAT_corrections.header.resolution;

            LIDAR_LOG_INFO <<m_PandarAT_corrections.start_frame[i] / 25600.f<<","<<m_PandarAT_corrections.end_frame[i] / 25600.f;
          } 

          for (int i = 0; i < 128; i++) {
            m_PandarAT_corrections.azimuth[i] =
                m_PandarAT_corrections.azimuth[i] *
                m_PandarAT_corrections.header.resolution;
            m_PandarAT_corrections.elevation[i] =
                m_PandarAT_corrections.elevation[i] *
                m_PandarAT_corrections.header.resolution;
          }
          for (int i = 0; i < (m_PandarAT_corrections.header.channel_number * 180); i++) {
            m_PandarAT_corrections.azimuth_offset[i] =
                m_PandarAT_corrections.azimuth_offset[i] *
                m_PandarAT_corrections.header.resolution;
            m_PandarAT_corrections.elevation_offset[i] =
                m_PandarAT_corrections.elevation_offset[i] *
                m_PandarAT_corrections.header.resolution;
          }

          return 0;
        } break;
        default:
          break;
      }
    }
    return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 1;
}

bool PointCloudParser::loadCorrectionFile(){
    LIDAR_LOG_TRACE << "load correction file from config ATD128P.dat now:";
    std::ifstream fin("/cfg/lidar_intrinsic_param/ATD128P.dat");
    if (fin.is_open()) {
      LIDAR_LOG_TRACE << "Open correction file success";
      int length = 0;
      // int target_length = 47176;
      std::string strlidarCalibration;
      fin.seekg(0, std::ios::end);              //设置 get 指针的位置，该指针确定下一个读取操作将开始的位置,std::ios::end用于指示应相对于文件末尾设置位置
      length = fin.tellg();                     //使用'seektellg()获取当前位置，该位置表示文件的大小
      if(length != target_length){
        LIDAR_LOG_ERROR << "Open correction file fail, length is:" << length;
        return false;
      }
      LIDAR_LOG_INFO << "Open correction file success, length is:" << length;
      fin.seekg(0, std::ios::beg);              //std::ios::beg用于指示应相对于文件的开头设置位置
      char *buffer = new char[length];
      if (buffer == nullptr) {
        LIDAR_LOG_ERROR << "Failed to allocate memory for buffer";
        fin.close();
        return false;
      }
      fin.read(buffer, length);
      fin.close();
      strlidarCalibration = buffer;
      int ret = loadCorrectionString(buffer);
      if (ret != 0) {
        LIDAR_LOG_TRACE << "Parse local Correction file Error";
        return false;
      }          
      // m_bGetCorrectionSuccess = true;
      LIDAR_LOG_TRACE << "Parse local Correction file Success!!!";
     
    } else {
      LIDAR_LOG_TRACE << "Open correction file failed";
      return false;
    }
    return true;
}


Eigen::Affine3f PointCloudParser::DrToAffine3f(std::shared_ptr<hozon::localization::Localization> proto){  
    Eigen::Quaternionf rotation(proto->pose_local().quaternion().w(),proto->pose_local().quaternion().x(),
                              proto->pose_local().quaternion().y(),proto->pose_local().quaternion().z());
    Eigen::Vector3f position(proto->pose_local().position().x(),proto->pose_local().position().y(),
                            proto->pose_local().position().z());                  
    LIDAR_LOG_TRACE << "success position().x() is: "<< position(0) << "  "<<position(1)<< "  "<<position(2)<< "  ";
    Eigen::Matrix3f rotation_matrix = rotation.toRotationMatrix();      // 步骤 1: 创建一个旋转矩阵
    Eigen::Affine3f affine_matrix = Eigen::Affine3f::Identity();        // 步骤 2: 创建一个仿射矩阵并设置旋转部分
    affine_matrix.linear() = rotation_matrix;
    affine_matrix.translation() = position;                             // 步骤 3: 设置仿射矩阵的平移部分

    return affine_matrix;
}

bool PointCloudParser::LoadExtrinsics(Eigen::Affine3f& ext) {

	std::string lidarExtrinsicsPara;
	hozon::netaos::cfg::ConfigParam::Instance()->GetParam("conf_calib_lidar/roof_lidar_params", lidarExtrinsicsPara);
	LIDAR_LOG_INFO << "lidarExtrinsicsPara is: "<< lidarExtrinsicsPara;
  if(lidarExtrinsicsPara.empty()){
    LIDAR_LOG_WARN << "lidarExtrinsicsPara is empty.";
    return false;
  }

	// 创建 JsonCpp 的根值
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::istringstream jsonStream(lidarExtrinsicsPara);
    Json::Reader reader;
    reader.parse(jsonStream, root);

    // 检查解析是否成功
    if (!reader.good()) {
        LIDAR_LOG_INFO << "Error parsing JSON:: " << reader.getFormattedErrorMessages();
        std::cerr << "Error parsing JSON: " << reader.getFormattedErrorMessages() << std::endl;
        return false;
    }

    
	// if(!root.isMember("date") || !root["euler_angle"].isMember("yaw") || root["rotation"].isMember("qw") || root["translation"].isMember("z")){
	// 	LIDAR_LOG_INFO << "Lidar extrinsics para has missing keys.";
    //     return false;
	// }
	// 访问 JSON 中的数据
    std::string child_frame_id = root["child_frame_id"].asString();
    std::string frame_id = root["frame_id"].asString();
    std::string lidar_sn = root["lidar_sn"].asString();
    std::string date = root["date"].asString();
    double pitch = root["euler_angle"]["pitch"].asDouble();
    double roll = root["euler_angle"]["roll"].asDouble();
    double yaw = root["euler_angle"]["yaw"].asDouble();
    double delta_pitch = root["euler_angle"]["delta_pitch"].asDouble();
    double delta_yaw = root["euler_angle"]["delta_yaw"].asDouble();
    double qw = root["rotation"]["qw"].asDouble();
    double qx = root["rotation"]["qx"].asDouble();
    double qy = root["rotation"]["qy"].asDouble();
    double qz = root["rotation"]["qz"].asDouble();
    double x = root["translation"]["x"].asDouble();
    double y = root["translation"]["y"].asDouble();
    double z = root["translation"]["z"].asDouble();
    int type = root["type"].asInt();

    // 打印提取的信息

    LIDAR_LOG_INFO << "date: " << date;
    LIDAR_LOG_INFO << "pitch: " << pitch;
    LIDAR_LOG_INFO << "roll: " << roll;
    LIDAR_LOG_INFO << "yaw: " << yaw;
    LIDAR_LOG_INFO << "qw: " << qw;
    LIDAR_LOG_INFO << "qx: " << qx;
    LIDAR_LOG_INFO << "qy: " << qy;
    LIDAR_LOG_INFO << "qz: " << qz;
    LIDAR_LOG_INFO << "x: " << x;
    LIDAR_LOG_INFO << "y: " << y;
    LIDAR_LOG_INFO << "z: " << z;

    ext = Eigen::Affine3f::Identity();
    Eigen::Quaternionf quater;
    quater.x() = qx;
    quater.y() = qy;
    quater.z() = qz;
    quater.w() = qw;

    yaw = yaw / 180 * M_PI;
    pitch = pitch / 180 * M_PI;
    roll = roll / 180 * M_PI;
    Eigen::Matrix3f R;                                              //创建旋转矩阵
    R = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

    Eigen::Quaternionf q(R);                                        //创建四元数
    Eigen::Translation3f translation(x, y, z);                      //创建平移变换
    // use euler angle
    ext = translation * q.toRotationMatrix();                       //将平移变化与旋转变换结合
    return true;
}


void PointCloudParser::motion_compensate(const std::shared_ptr<hozon::ethstack::lidar::PointCloudFrame> send_data_ptr_,
        const Eigen::Affine3f& start_pose, const Eigen::Affine3f& end_pose,
        const Eigen::Affine3f& lidar_extrinsics,int block_num) {
    std::vector<Eigen::Affine3f> motion_poses;
    Eigen::Vector3f translation =                                     //计算两个位姿之间的平移信息
        start_pose.translation() - end_pose.translation();
    Eigen::Quaternionf q_max(end_pose.linear());                      //表示停止姿态的四元数
    Eigen::Quaternionf q_min(start_pose.linear());
    Eigen::Quaternionf q1(q_max.conjugate() * q_min);                 //计算两个姿态之间的相对旋转，用四元数表示
    Eigen::Quaternionf q0(Eigen::Quaternionf::Identity());            //单位四元数
    q1.normalize();                                                   //对q1进行归一化处理  
    translation = q_max.conjugate() * translation;                    //将平移矢量应用于最大姿态的共轭旋转
    float d = q0.dot(q1);                                             //计算q0与q1的点积
    float abs_d = abs(d);
    float theta = acos(abs_d);                                        //theta表示旋转角度
    float sin_theta = sin(theta);
    float c1_sign = (d > 0.0) ? 1.0 : -1.0;                           //判断正负
    for (int j = 0; j < block_num; ++j) {                             //按列遍历
      float t = static_cast<float>(block_num - j) / block_num;        //列数越大，t越小，t 表示从 0 到 1 的插值参数，可以表示补偿比例
      Eigen::Translation3f ti(t * translation);                       //表示当前阶段 t 对应的平移变换
      float c0 = sin((1.0 - t) * theta) / sin_theta;                  //计算插值参数 t 对应的旋转插值系数                 
      float c1 = sin(t * theta) / sin_theta * c1_sign;                //c0 和 c1 是用于线性插值旋转四元数 q0 和 q1 的系数
      Eigen::Quaternionf qi(c0 * q0.coeffs() + c1 * q1.coeffs());     //qi表示当前阶段 t 对应的旋转四元数
      Eigen::Affine3f trans = Eigen::Affine3f::Identity();            //存储运动补偿后的位姿
      if (abs_d < 1.0 - 1.0e-8) {                                     //根据 abs_d 的值，选择不同的运动补偿方式
        trans = ti * qi * lidar_extrinsics;                           //旋转变化不大时，将平移变换 ti 应用到旋转后的位姿 qi 上
      } else {
        trans = ti * lidar_extrinsics;
      }
      motion_poses.push_back(trans);                                  //得到的运动补偿后的位姿 trans 被添加到motion_poses向量中
    }

    for (unsigned long i = 0; i < send_data_ptr_->data.size(); ++i) {
      const auto &pt = send_data_ptr_->data[i];
      Eigen::Vector3f vec(pt.x, pt.y, pt.z);
      vec = motion_poses[send_data_ptr_->data[i].block] * vec;
      send_data_ptr_->data[i].x = vec(0);
      send_data_ptr_->data[i].y = vec(1);
      send_data_ptr_->data[i].z = vec(2);
    }
}

void PointCloudParser::SaveNoMotionCompensate(const std::shared_ptr<hozon::ethstack::lidar::PointCloudFrame> send_data_ptr,
                                          const Eigen::Affine3f& lidar_extrinsics){
    pcl::PointCloud<pcl::PointXYZI>::Ptr right_cloud_pcl(new pcl::PointCloud<pcl::PointXYZI>);
    double lidar_ts;
    for (uint32_t i = 0; i < send_data_ptr->data.size(); i++) {
        const auto& pt = send_data_ptr->data[i];
        Eigen::Vector3f original_point(pt.x, pt.y, pt.z);
        Eigen::Vector3f transformed_point = lidar_extrinsics * original_point;
        pcl::PointXYZI tmp_pt;
        tmp_pt.x = transformed_point.x();
        tmp_pt.y = transformed_point.y();
        tmp_pt.z = transformed_point.z();
        tmp_pt.intensity = pt.intensity;
        right_cloud_pcl->points.push_back(tmp_pt);
        lidar_ts = static_cast<double>(send_data_ptr->header.stamp.sec) + double(send_data_ptr->header.stamp.nsec) / LIDAR_1SEC_NSEC;
    }
    right_cloud_pcl->width = right_cloud_pcl->points.size();
    right_cloud_pcl->height = 1;

    // save image
    // std::string result_image_path_ = "/data/ch_work_sapce/pointcloud_pb";               //x86   
    std::string result_image_path_ = "/opt/usr/ch/no_motion_compensate";
    std::string save_path = result_image_path_ + "/" + std::to_string(lidar_ts);
    // std::string cmd_str = "mkdir -p " + save_path;
    // system(cmd_str.c_str());

    pcl::PCDWriter writer;

    writer.write(save_path + ".pcd", *right_cloud_pcl, true);
}





}
}
}
