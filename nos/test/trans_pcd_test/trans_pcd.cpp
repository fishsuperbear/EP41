// 编译x86版本
#include <iostream>
#include <fstream>
#include "cm/include/proto_cm_reader.h"
#include "log/include/default_logger.h"
#include <unistd.h>
#include "proto/soc/point_cloud.pb.h"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/io/pcd_io.h"

#include <bits/time.h>
#include <bits/types/struct_timespec.h>
#include "yaml-cpp/yaml.h"


static uint16_t frames_ = 0;


Eigen::Affine3f LoadExtrinsics(const std::string& extrinsics_path) {
    Eigen::Affine3f ext = Eigen::Affine3f::Identity();

    YAML::Node ext_file = YAML::LoadFile(extrinsics_path);    //将yaml文件导出到ext_file变量

    float qw, qx, qy, qz;                     //用四元数表示旋转矩阵
    float x, y, z;                            //xyz表示平移矢量

    qw = ext_file["rotation"]["qw"].as<float>();
    qx = ext_file["rotation"]["qx"].as<float>();
    qy = ext_file["rotation"]["qy"].as<float>();
    qz = ext_file["rotation"]["qz"].as<float>();
    x = ext_file["translation"]["x"].as<float>();
    y = ext_file["translation"]["y"].as<float>();
    z = ext_file["translation"]["z"].as<float>();

    ext = Eigen::Affine3f::Identity();
    Eigen::Quaternionf quater;
    quater.x() = qx;
    quater.y() = qy;
    quater.z() = qz;
    quater.w() = qw;

    float yaw, pitch, roll;
    yaw = ext_file["euler_angle"]["yaw"].as<float>();               //偏航角
    pitch = ext_file["euler_angle"]["pitch"].as<float>();           //俯仰角
    roll = ext_file["euler_angle"]["roll"].as<float>();             //横滚角
    yaw = yaw / 180 * M_PI;
    pitch = pitch / 180 * M_PI;
    roll = roll / 180 * M_PI;
    Eigen::Matrix3f R;                                              //创建旋转矩阵
    R = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

    Eigen::Quaternionf q(R);                                        //创建四元数
    Eigen::Translation3f translation(x, y, z);                      //创建平移变换
    ext = translation * q.toRotationMatrix();                       //将平移变化与旋转变换结合

    return ext;
}


void trans(pcl::PointCloud<pcl::PointXYZI>::Ptr right_cloud_pcl,Eigen::Affine3f ext) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr dest_pcd(new pcl::PointCloud<pcl::PointXYZI>);
    for (auto& point : right_cloud_pcl->points) {
        Eigen::Vector3f original_point(point.x, point.y, point.z);
        Eigen::Vector3f transformed_point = ext * original_point;
        pcl::PointXYZI tmp_pt;
        tmp_pt.x = transformed_point.x();
        tmp_pt.y = transformed_point.y();
        tmp_pt.z = transformed_point.z();
        tmp_pt.intensity = point.intensity;
        dest_pcd->points.push_back(tmp_pt);
    }
    dest_pcd->width = dest_pcd->points.size();
    dest_pcd->height = 1;

    std::string result_image_path_ = "/data/ch_work_sapce/trans_pcd";
    std::string save_path = result_image_path_ + "/";
    pcl::PCDWriter writer;
    writer.write(save_path + "1.pcd", *dest_pcd, true);
    // return 0;
}


int main(int argc, char* argv[]) {
// 定义YAML文件路径
    std::string yamlFilePath = "/data/ch_work_sapce/trans_pcd/lidar_extrinsics.yaml";
    std::cout<<"---------------------1-----"<<std::endl;
    // 读取外参矩阵
    Eigen::Affine3f ext = LoadExtrinsics(yamlFilePath);
    std::cout<<"---------------------2-----"<<std::endl;
    // 定义PCD文件夹路径
    std::string pcdPath = "/data/ch_work_sapce/trans_pcd/2524609740.597227.pcd";
    std::cout<<"---------------------3-----"<<std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr right_cloud_pcl(new pcl::PointCloud<pcl::PointXYZI>);
    std::cout<<"---------------------4-----"<<std::endl;
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcdPath, *right_cloud_pcl);
    std::cout<<"---------------------5-----"<<std::endl;
    trans(right_cloud_pcl,ext);
    std::cout<<"---------------------6-----"<<std::endl;
    return 0;
}