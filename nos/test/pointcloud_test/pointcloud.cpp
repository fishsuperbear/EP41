// 编译x86版本
#include "cm/include/proto_cm_reader.h"
#include "log/include/default_logger.h"
#include <unistd.h>
#include <string>
#include "proto/soc/point_cloud.pb.h"

#include "logprecision.h"
#include "logfixed.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/io/pcd_io.h"

#include <bits/time.h>
#include <bits/types/struct_timespec.h>
#include <iostream>

#define  CLOCK_VIRTUAL  12u

inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return static_cast<double>(time_now.tv_sec)
             + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
}  

inline double GetAbslTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_VIRTUAL, &time_now);

    return static_cast<double>(time_now.tv_sec)
             + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
}  

static uint16_t frames_ = 0;

void cb(std::shared_ptr<hozon::soc::PointCloud> msg) {

    // DF_LOG_INFO << "lidar pub_time is: " << std::to_string(msg->header().publish_stamp());
    // DF_LOG_INFO << "lidar recv_time is: " <<std::to_string(GetRealTimestamp());
    DF_LOG_INFO << "lidar pub_time is: " << hozon::netaos::log::Fixed()  << SET_PRECISION(9) << msg->header().publish_stamp();
    DF_LOG_INFO << "lidar recv_time is: " <<  hozon::netaos::log::Fixed() << SET_PRECISION(9) << GetRealTimestamp();
    pcl::PointCloud<pcl::PointXYZI>::Ptr right_cloud_pcl(new pcl::PointCloud<pcl::PointXYZI>);
    std::string lidar_ts;
    // std::cout << "&abc" << &"111" << std::endl;
    std::stringstream oss;
    
    //Hesai lidar
    DF_LOG_INFO << "hesai lidar width is: " << msg->points_size();
    for (uint32_t i = 0; i < msg->points_size(); i++) {
        const auto& pt = msg->points(i);
        pcl::PointXYZI tmp_pt;
        tmp_pt.x = pt.x();
        tmp_pt.y = pt.y();
        tmp_pt.z = pt.z();
        tmp_pt.intensity = pt.intensity();
        right_cloud_pcl->points.push_back(tmp_pt);
        
    }
    // lidar_ts = static_cast<double>(msg->header().sensor_stamp().lidar_stamp());
    oss << std::fixed << std::setprecision(9) << msg->header().sensor_stamp().lidar_stamp();
    lidar_ts = oss.str();

    right_cloud_pcl->width = right_cloud_pcl->points.size();
    right_cloud_pcl->height = 1;

    // save image
    // std::string result_image_path_ = "/data/ch_work_sapce/pointcloud_pb";               //x86   
    std::string result_image_path_ = "/opt/usr/ch/motion_compensate";
    std::string save_path = result_image_path_ + "/" + lidar_ts;
    // std::string cmd_str = "mkdir -p " + save_path;
    // system(cmd_str.c_str());

    pcl::PCDWriter writer;

    writer.write(save_path + ".pcd", *right_cloud_pcl, true);
    frames_++;
    DF_LOG_INFO << "save frames: " << frames_;
    // return 0;
}

int main(int argc, char* argv[]) {
    hozon::netaos::cm::ProtoCMReader<hozon::soc::PointCloud> reader;
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret = reader.Init(0, "/soc/pointcloud", cb);

    sleep(20);

    reader.Deinit();
    DF_LOG_INFO << "Deinit end." << ret;
}