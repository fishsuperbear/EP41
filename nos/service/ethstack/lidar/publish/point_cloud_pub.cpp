#include <cstdint>
#include <iostream>
#include <thread>
#include <memory>
#include <unistd.h>
#include <pthread.h>
#include "point_cloud_pub.h"
#include "cfg/include/config_param.h"
#include "common/logger.h"
#include "faultmessage/lidar_fault_report.h"
#include "faultmessage/lidar_status_report.h"
#include "proto/soc/point_cloud.pb.h"

namespace hozon {
namespace ethstack {
namespace lidar {

static std::mutex inst_mutex_;
PointCloudPub* PointCloudPub::instance_ = nullptr;
uint32_t flag  = 0;


PointCloudPub& PointCloudPub::Instance() {
  if (nullptr == instance_) {
    std::lock_guard<std::mutex> lck(inst_mutex_);
    if (nullptr == instance_) {
      instance_ = new PointCloudPub();
    }
  }
  return *instance_;
}

PointCloudPub::PointCloudPub()
: serving_(false)
, statusReport_(false) {

}

PointCloudPub::~PointCloudPub() {

}

void PointCloudPub::Init(const std::string& topic) {

    int32_t ret = writer.Init(0, topic);
    if (ret < 0) {
        LIDAR_LOG_INFO << "Fail to init writer " << ret;
        LidarFaultReport::Instance().ReportAbstractFault(AbstractFaultObject::CM_INIT_ERROR, CURRENT_FAULT);
    }
    serving_ = true;
    statusReport_ = true;
    LIDAR_LOG_INFO << "PointCloudPub Init Complete.";
}

void PointCloudPub::Pub() {
    std::thread th([&]()-> void {
        auto pred = [this]() -> bool {
            return (send_data_vec_.size() > 0);
        };
        LIDAR_LOG_INFO << "PointCloudPub Pub Thread Create.";
        while (statusReport_) {                                              //需要在初始化阶段执行：PointCloudPub::Init（）函数
            std::unique_lock<std::mutex> lock(data_mutex_);
            if (send_data_cv_.wait_for(lock, std::chrono::milliseconds(20), pred)) {
                for (auto dataptr : send_data_vec_) {
                    dataptr->header.frameId = lidar_frame_id_;
                    hozon::soc::PointCloud pointcloud_proto;
                    double lidar_time = double(dataptr->header.stamp.sec) + double(dataptr->header.stamp.nsec) / LIDAR_1SEC_NSEC;
                    LIDAR_LOG_TRACE << "Pub before points size is: "<< pointcloud_proto.points_size(); 
                    for (auto point : dataptr->data) {
                        // 保存点云信息到算法的proto中
                        auto add = pointcloud_proto.add_points();
                        add->set_x(point.x);
                        add->set_y(point.y);
                        add->set_z(point.z);
                        add->set_time(point.time);
                        add->set_distance(point.distance);
                        add->set_pitch(point.pitch);
                        add->set_yaw(point.yaw);
                        add->set_intensity(point.intensity);
                        add->set_ring(point.ring);
                        add->set_block(point.block);
                    }
                    pointcloud_proto.set_is_valid(1);
                    pointcloud_proto.set_is_big_endian(dataptr->isBigEndian);
                    pointcloud_proto.set_height(dataptr->height);
                    pointcloud_proto.set_width(dataptr->width);
                    pointcloud_proto.set_point_step(dataptr->pointStep);
                    pointcloud_proto.set_row_step(dataptr->rowStep);
                    pointcloud_proto.set_is_dense(dataptr->isDense);
                    pointcloud_proto.set_ecu_serial_number("hesai-AT128P");
                    pointcloud_proto.mutable_header()->mutable_sensor_stamp()->set_lidar_stamp(lidar_time);
                    double pub_time = GetRealTimestamp();
                    pointcloud_proto.mutable_header()->set_publish_stamp(pub_time);
                    pointcloud_proto.mutable_header()->set_seq(dataptr->header.seq);

                    LIDAR_LOG_TRACE << "Pub after points size is: "<< pointcloud_proto.points_size(); 
                    uint8_t ret = writer.Write(pointcloud_proto);
                    LIDAR_LOG_INFO << "pointcloud_proto size is: "<< pointcloud_proto.ByteSizeLong();
                    if (ret < 0) {
                        LIDAR_LOG_ERROR << "Fail to write " << ret;
                    }
                    dataptr->data.clear();
                    LIDAR_LOG_INFO << "Send point cloud frame ok. frameid[" << flag << "].";
                    flag++;
                    report_status_cv_.notify_all();
                    
                    LIDAR_LOG_TRACE << "width is: "<< dataptr->width << "height is: "<< dataptr->height << "point_step is: "<< dataptr->pointStep<< "row_step is: "<< dataptr->rowStep;
                }
                send_data_vec_.clear();
            }
        }
    });
    pthread_setname_np(th.native_handle(), (std::string("Pub")).c_str());
    th.detach();    
}

void PointCloudPub::lidarStatusReportFunction() {
    std::thread th([&]()-> void {
        LIDAR_LOG_INFO << "PointCloudPub lidarStatusReportFunction Thread Create.";
        while (statusReport_) { 
            std::unique_lock<std::mutex> lock(status_mutex_);
            report_status_cv_.wait_for(lock, std::chrono::milliseconds(5000));
            uint8_t lidar_history_status = hozon::ethstack::lidar::LidarStatusReport::Instance().GetLidarStatus();
            uint8_t lidar_current_status = 1;
            if (lidar_history_status != lidar_current_status) {
                hozon::netaos::cfg::ConfigParam::Instance()->SetParam("system/lidar_status", lidar_current_status);
                LIDAR_LOG_TRACE << "success write lidar work status to cfg.";
                hozon::ethstack::lidar::LidarStatusReport::Instance().SetLidarStatus(lidar_current_status);
                LIDAR_LOG_TRACE << "success write lidar work status to LidarStatus class.";
            }
            
        }
    });
    pthread_setname_np(th.native_handle(), (std::string("lidarStatusReport")).c_str());
    th.detach();
}



void PointCloudPub::Stop() {
    // 停止服务
    serving_ = false;
    statusReport_ = false;
    send_data_cv_.notify_all();
    report_status_cv_.notify_all();
    writer.Deinit();
        
}

std::shared_ptr<PointCloudFrame> PointCloudPub::GetSendDataPtr() {
    auto ptr = std::make_shared<PointCloudFrame>();
    if (ptr == nullptr) {
        LIDAR_LOG_ERROR << " PointCloudFrame data_ptr create failed.";
        return nullptr;
    }
    ptr->header.stamp = {0, 0};
    return ptr;
}

void PointCloudPub::SetSendData(std::shared_ptr<PointCloudFrame> dataptr) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    send_data_vec_.push_back(dataptr);
    if (send_data_vec_.size() > 5) {
        LIDAR_LOG_WARN << "----------Warinning: Send buffer over 5 frames: " << send_data_vec_.size() << "----------";
    }
    send_data_cv_.notify_all();
}


}
}
}
