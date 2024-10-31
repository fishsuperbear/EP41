#include "cm/include/proto_cm_writer.h"
#include "proto/test/soc/for_test.pb.h"
#include "proto/dead_reckoning/dr.pb.h"
#include "log/include/default_logger.h"
#include <unistd.h>


int main(int argc, char* argv[]) {
    hozon::netaos::cm::ProtoCMWriter<hozon::dead_reckoning::DeadReckoning> writer;
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret = writer.Init(0, "/localization/deadreckoning");
    if (ret < 0) {
        DF_LOG_ERROR << "Fail to init writer " << ret;
        return -1;
    }
    double i = 0;
    while(1){
        hozon::dead_reckoning::DeadReckoning msg;
        msg.mutable_pose()->mutable_pose_local()->mutable_position()->set_x(0);
        msg.mutable_pose()->mutable_pose_local()->mutable_position()->set_y(0);
        msg.mutable_pose()->mutable_pose_local()->mutable_position()->set_z(0);
        msg.mutable_pose()->mutable_pose_local()->mutable_quaternion()->set_x(0);
        msg.mutable_pose()->mutable_pose_local()->mutable_quaternion()->set_y(0);
        msg.mutable_pose()->mutable_pose_local()->mutable_quaternion()->set_z(0);
        msg.mutable_pose()->mutable_pose_local()->mutable_quaternion()->set_w(1);

        ret = writer.Write(msg);
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to write " << ret;
        }
        
        // sleep(0.01);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        DF_LOG_INFO << "success to write " << i;
        i = i + 0.2;
    }

    writer.Deinit();
    DF_LOG_INFO << "Deinit end." << ret;
}