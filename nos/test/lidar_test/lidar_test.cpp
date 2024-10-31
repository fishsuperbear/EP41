// 编译x86版本
#include "cm/include/proto_cm_reader.h"
#include "log/include/default_logger.h"
#include <unistd.h>
#include "proto/soc/point_cloud.pb.h"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/io/pcd_io.h"

#include <bits/time.h>
#include <bits/types/struct_timespec.h>

#define  CLOCK_VIRTUAL  12u

inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return static_cast<double>(time_now.tv_sec)
             + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
}  


static uint16_t frames_ = 0;
static uint16_t print_flag = 0;
double start_time;
double end_time;

void cb(std::shared_ptr<hozon::soc::PointCloud> msg) {

    if (frames_ == 0) {
        start_time = msg->header().publish_stamp();
    }
    if (frames_ == 10) {
        end_time = msg->header().publish_stamp();
    }

    if (frames_ >= 10) {
        if((end_time - start_time <= 1.5) && (print_flag == 0)){
            std::cout<<"!!!!!hz_test_success!!!!!"<<std::endl;
            print_flag++;
            return;
        }
        return;
    }
    frames_++;
    // std::cout<<"frame is: "<<frames_<<std::endl;
}

int main(int argc, char* argv[]) {
    hozon::netaos::cm::ProtoCMReader<hozon::soc::PointCloud> reader;
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret = reader.Init(0, "/soc/pointcloud", cb);
    sleep(5);

    if(frames_ == 0){
        std::cout<<"!!!!!hz_test_failed!!!!!"<<std::endl;
    }
    reader.Deinit();
}