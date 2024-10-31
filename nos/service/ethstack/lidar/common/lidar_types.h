#ifndef LIDAR_DATADEF_LIDAR_TYPES_H
#define LIDAR_DATADEF_LIDAR_TYPES_H

#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace hozon {
namespace ethstack {
namespace lidar {

template <typename T>
struct Point3D {
  T x = 0;
  T y = 0;
  T z = 0;
  Point3D() : x(0), y(0), z(0) {}
  Point3D(const T &valueX, const T &valueY, const T &valueZ)
      : x(valueX), y(valueY), z(valueZ) {}
  Point3D(const Point3D<T> &pt) : x(pt.x), y(pt.y), z(pt.z) {}
};

using float32_t = float;
using float64_t = double;
using float128_t = long double;

/* ******************************************************************************
    功能描述       :  Lidar输出的点
******************************************************************************
*/
struct PointXYZI : public Point3D<float32_t> {
  uint16_t intensity;
  PointXYZI() : Point3D<float32_t>(), intensity(0U) {}
};

struct Time {
  ::uint32_t sec;
  ::uint32_t nsec;
};

/* ******************************************************************************
    结构 名        :  LidarFrame
    功能描述       :  Lidar每帧输出的数据
******************************************************************************
*/
template <typename T>
struct LidarFrame {
  Time timestamp;
  uint32_t seq;
  std::string frameID;
  bool isValid = false;
  std::vector<T> pointCloud;
};


///  data struct defination
typedef struct EthernetSocketInfo {
    std::string frame_id;
    std::string if_name;
    std::string local_ip;
    std::string remote_ip;
    std::string multicast;
    uint16_t local_port;
    uint16_t remote_port;

} EthernetSocketInfo;

typedef struct EthernetPacket {
    uint32_t sec;
    uint32_t nsec;
    uint32_t len;
    uint8_t data[3690] = { 0 };
} EthernetPacket;


using PointCloudFrameXYZI = LidarFrame<PointXYZI>;

using DataPtr = std::shared_ptr<uint8_t>;
#define MakeDataBuf(size)                             \
  std::shared_ptr<uint8_t> buf_ptr(new uint8_t[size], \
                                   [](uint8_t *p) { delete[] p; })
using DataHandler = std::function<void(std::shared_ptr<uint8_t>, int)>;

#define FRAME_ID_LIDAR_1 "lidar1"
#define FRAME_ID_LIDAR_2 "lidar2"

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // ETHSTACK_LIDAR_DATADEF_LIDAR_TYPES_H
