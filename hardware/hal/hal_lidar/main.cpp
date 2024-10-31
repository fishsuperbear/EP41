#include <unistd.h>

#include "hal_lidar.hpp"

#ifdef PCL_SHOW
#include <pcl/point_types.h>

#include "pcl/io/pcd_io.h"
#include "pcl/io/ply_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/visualization/cloud_viewer.h"

pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
pcl::visualization::PCLVisualizer viewer_;

void pcl_show(std::vector<hal::lidar::PointXYZIT> cld) {
    for (int i = 0; i < cld.size(); i++) {
        hal::lidar::PointXYZIT point = cld[i];
        if (point.x == 0 && point.y == 0 && point.z == 0) {
            continue;
        }

        pcl::PointXYZI p;
        p.x = point.x;
        p.y = point.y;
        p.z = point.z;
        p.intensity = point.intensity;
        cloud_->points.push_back(p);
    }
    cloud_->width = cloud_->points.size();
    cloud_->height = 1;

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> single_color(cloud_, "intensity");
    viewer_.updatePointCloud<pcl::PointXYZI>(cloud_, single_color, "Sample Points 1");
    if (!viewer_.wasStopped()) {
        viewer_.spinOnce(0);
    }

    cloud_->points.clear();
}
#endif

void onPointsCallback(const hal::lidar::PointsCallbackData& data) {
    printf("receive plitclouds from lidar%d, seq: %lu, timestamp: %lu, size: %lu\n",
           data.index, data.sequence, data.timestamp, data.points.size());
#ifdef PCL_SHOW
    pcl_show(data.points);
#endif
}

int main(int argc, char** argv) {
#ifdef PCL_SHOW
    cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    // 背景设置为黑色
    viewer_.setBackgroundColor(0, 0, 0);
    viewer_.addPointCloud<pcl::PointXYZI>(cloud_, "Sample Points 1");
    viewer_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Sample Points 1");
    viewer_.addCoordinateSystem(1.0);
    viewer_.initCameraParameters();
#endif

    std::vector<hal::lidar::ConfigInfo> configs;

    hal::lidar::ConfigInfo config1;
    config1.index = 1;
    config1.model = "hesai_at128";
    config1.ip = "192.168.3.201";
    config1.port = 2368;
    configs.push_back(config1);

    hal::lidar::ConfigInfo config2;
    config2.index = 2;
    config2.model = "hesai_at128";
    config2.ip = "192.168.3.202";
    config2.port = 2369;
    configs.push_back(config2);

    // while (true) {
    hal::lidar::Start(configs, onPointsCallback);
    sleep(1000000);
    hal::lidar::Stop();

    //     sleep(2);
    // }

    return 0;
}
