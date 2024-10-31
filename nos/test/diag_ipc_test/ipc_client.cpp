#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>


#include "diag/ipc/api/ipc_client.h"
#include "diag/ipc/common/ipc_def.h"
#include "diag/ipc/proto/diag.pb.h"

#include "log/include/logging.h"



int main(int argc, char ** argv) {

    hozon::netaos::log::InitLogging("HZ_TEST",                                                             // the id of application
                                        "HZ_log application",                                                  // the log id of application
                                        hozon::netaos::log::LogLevel::kDebug,                                                      //the log level of application
                                        hozon::netaos::log::HZ_LOG2CONSOLE + hozon::netaos::log::HZ_LOG2FILE,  //the output log mode
                                        "./",                                                                  //the log file directory, active when output log to file
                                        10,                                                                    //the max number log file , active when output log to file
                                        10                                                                     //the max size of each  log file , active when output log to file
    );


    std::unique_ptr<hozon::netaos::diag::IPCClient> client = std::make_unique<hozon::netaos::diag::IPCClient>();

    auto res1 = client->Init("service_name_8");
    std::cout << "Init() : " << res1 << std::endl;

    // 转换为 Protocol Buffers 消息
    UdsDataMethod protoData;
    protoData.mutable_meta_info()->insert({{"key1", "value1"}, {"key2", "value2"}});
    protoData.set_sid(1);
    protoData.set_subid(2);
    protoData.set_resp_ack(3);
    protoData.set_data_len(4); 
    protoData.set_data_vec({10, 20, 30, 40, 50});

    // 序列化为字节流
    std::string serializedData = protoData.SerializeAsString();
    std::cout << "SerializeToArray() : " << serializedData << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<uint8_t> req (serializedData.begin(), serializedData.end());
    std::vector<uint8_t> resp;
    auto res2 = client->Request(req, resp, 2000);
    std::cout << "Request1() : " << res2 << std::endl;


    std::string str(resp.begin(), resp.end());
    // 反序列化为 Protocol Buffers 消息
    UdsDataMethod deserializedProtoData;
    deserializedProtoData.ParseFromString(str);
    std::cout << "ParseFromArray() : " << str << std::endl;

    return 0;
}
