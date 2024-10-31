#include <signal.h>
#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>

#include "idl/generated/avm_methodPubSubTypes.h"
#include "idl/generated/chassis_methodPubSubTypes.h"
#include "log/include/default_logger.h"
#include "cm/include/method.h"

using namespace hozon::netaos::cm;

std::mutex mtx;
std::condition_variable cv;
int g_stopFlag = 0;

class UserServer : public Server<avm_method, chassis_method> {
   public:
    UserServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data) : Server(req_data, resp_data) {}
    int32_t Process(const std::shared_ptr<avm_method> req, std::shared_ptr<chassis_method> resp) {
        std::cout << "hello world" << std::endl;
        return 0;
    }
};

void SigHandler(int signum) {
    g_stopFlag = 1;
    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_all();
}

int main(int argc, char** argv) {
    DefaultLogger::GetInstance().InitLogger();
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    std::shared_ptr<avm_methodPubSubType> req_data_type = std::make_shared<avm_methodPubSubType>();
    std::shared_ptr<chassis_methodPubSubType> resp_data_type = std::make_shared<chassis_methodPubSubType>();

    UserServer user_server(req_data_type, resp_data_type);

    // user_server.RegisterProcess(std::bind(&UserServer::Process, &user_server, std::placeholders::_1, std::placeholders::_2));

    user_server.Start(2, "test");

    while (!g_stopFlag) {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck);
    }
    user_server.Stop();

    return 0;
}
