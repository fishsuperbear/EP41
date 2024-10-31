#include "cm/include/proto_cm_writer.h"
#include "proto/test/soc/for_test.pb.h"
#include "log/include/default_logger.h"
#include <unistd.h>

inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1000 / 1000 / 1000;
}

int main(int argc, char* argv[]) {
    hozon::netaos::cm::ProtoCMWriter<adf::lite::dbg::WorkflowResult> writer;
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret = writer.Init(0, "workresult_pb");
    if (ret < 0) {
        DF_LOG_ERROR << "Fail to init writer " << ret;
        return -1;
    }

    for (int i = 0; i < 15; ++i) {
        adf::lite::dbg::WorkflowResult msg;
        msg.set_val1(i);
        msg.mutable_header()->set_seq(i);
        double timestamp = GetRealTimestamp();
        DF_LOG_INFO << "send timestamp is " << timestamp;
        msg.mutable_header()->set_publish_stamp(timestamp);


        ret = writer.Write(msg);
        if (ret < 0) {
            DF_LOG_ERROR << "Fail to write " << ret;
        }
        
        sleep(1);
    }

    writer.Deinit();
    DF_LOG_INFO << "Deinit end." << ret;
}