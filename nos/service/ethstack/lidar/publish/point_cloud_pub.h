#ifndef LIDAR_POINT_CLOUD_PUB_H
#define LIDAR_POINT_CLOUD_PUB_H

#include <condition_variable>
#include <mutex>
#include <thread>
#include <bits/time.h>
#include <bits/types/struct_timespec.h>
#include <atomic>

#include "cm/include/skeleton.h"
#include "idl/generated/avmPubSubTypes.h"
#include "common/logger.h"
#include "protocol/point_cloud.h"

// #include "pcl/point_cloud.h"
// #include "pcl/point_types.h"
// #include "pcl/io/pcd_io.h"

#include "cm/include/proto_cm_writer.h"
#include "proto/soc/point_cloud.pb.h"
#include "faultmessage/lidar_fault_report.h"
#include "cfg/include/config_param.h"
#include "faultmessage/lidar_status_report.h"

namespace hozon {
namespace ethstack {
namespace lidar {


const double LIDAR_1SEC_NSEC = 1000000000.0;

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

class PointCloudPub {
 public:
  static PointCloudPub& Instance();
  virtual ~PointCloudPub();

  void Init(const std::string& topic);
  void Pub();
  void run();
  void Stop();
  void lidarStatusReportFunction();

  std::shared_ptr<PointCloudFrame> GetSendDataPtr();
  void SetSendData(std::shared_ptr<PointCloudFrame> dataptr);

  int frames_ = 0;
  int max_save_nums_ = 100;



 private:
  PointCloudPub();
  hozon::netaos::cm::ProtoCMWriter<hozon::soc::PointCloud> writer;

  bool serving_;
  bool statusReport_;
  std::string lidar_frame_id_; 
  std::vector<std::shared_ptr<PointCloudFrame>> send_data_vec_;
  std::condition_variable send_data_cv_;
  std::condition_variable report_status_cv_;
  std::mutex data_mutex_;
  std::mutex status_mutex_;
  static PointCloudPub* instance_;
};

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon

#endif
