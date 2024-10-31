#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

struct LidarPoint {
    PCL_ADD_POINT4D  // 添加pcl里xyz
        uint32_t time;
    double distance;
    double pitch;
    double yaw;
    float intensity;
    uint32_t ring;
    uint32_t block;
    uint32_t label;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are
                                     // aligned,确保定义新类型点云内存与SSE对齐
} EIGEN_ALIGN16;                     // 强制SSE填充以正确对齐内存

POINT_CLOUD_REGISTER_POINT_STRUCT(LidarPoint, (float, x, x)(float, y, y)(float, z, z)(std::uint32_t, time, time)(double, distance, distance)(double, pitch, pitch)(double, yaw, yaw)(
                                                  float, intensity, intensity)(std::uint32_t, ring, ring)(std::uint32_t, block, block)(std::uint32_t, label, label))

typedef LidarPoint LPoint;
typedef pcl::PointCloud<LPoint> PPointCloud;
