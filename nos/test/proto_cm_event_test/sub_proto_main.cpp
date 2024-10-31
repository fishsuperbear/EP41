#include "cm/include/proto_cm_reader.h"
#include "proto/test/soc/for_test.pb.h"
#include "log/include/default_logger.h"
#include <unistd.h>

void cb(std::shared_ptr<adf::lite::dbg::WorkflowResult> msg) {
    DF_LOG_INFO << "msg valL: " << msg->val1();
}

int main(int argc, char* argv[]) {
    hozon::netaos::cm::ProtoCMReader<adf::lite::dbg::WorkflowResult> reader;
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret = reader.Init(0, "workresult_pb", cb);

    sleep(20);

    reader.Deinit();
    DF_LOG_INFO << "Deinit end." << ret;
}