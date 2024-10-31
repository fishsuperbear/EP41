#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>

#include "log/include/logging.h"
#include "diag/ipc/api/ipc_server.h"
#include "diag/ipc/common/ipc_def.h"
#include "diag/ipc/proto/diag.pb.h"


class TestServer : public hozon::netaos::diag::IPCServer
{
    public:
        virtual int32_t Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp)
        {
            std::cout << "TestServer Process" << std::endl;
            std::cout << "client req is:" << std::endl;

            std::string str(req.begin(), req.end());
            // 反序列化为 Protocol Buffers 消息
            UdsDataMethod deserializedProtoData;
            deserializedProtoData.ParseFromString(str);
            std::cout << "ParseFromArray() : " << str << std::endl;


            // 转换为 Protocol Buffers 消息
            UdsDataMethod protoData;
            protoData.mutable_meta_info()->insert({{"key3", "value4"}, {"key5", "value6"}});
            protoData.set_sid(11);
            protoData.set_subid(22);
            protoData.set_resp_ack(33);
            protoData.set_data_len(44); 
            protoData.set_data_vec({11, 22, 33, 44, 55});

            // 序列化为字节流
            std::string serializedData = protoData.SerializeAsString();
            std::cout << "SerializeToArray() : " << serializedData << std::endl;

            std::vector<uint8_t> resp11 (serializedData.begin(), serializedData.end());

            resp = resp11;

            return 0;
        }

    private:

};

int main(int argc, char ** argv) {

    hozon::netaos::log::InitLogging("HZ_TEST",                                                             // the id of application
                                    "HZ_log application",                                                  // the log id of application
                                    hozon::netaos::log::LogLevel::kDebug,                                                      //the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE + hozon::netaos::log::HZ_LOG2FILE,  //the output log mode
                                    "./",                                                                  //the log file directory, active when output log to file
                                    10,                                                                    //the max number log file , active when output log to file
                                    10                                                                     //the max size of each  log file , active when output log to file
    );
    std::string serverName{"service_name_8"};
    std::unique_ptr<TestServer> server = std::make_unique<TestServer>();
    server->Start(serverName);

    std::this_thread::sleep_for(std::chrono::seconds(10));

    server->Stop();

    return 0;
}
