#include <cstdint>
#include <iostream>
#include "server.h"
#include <yaml-cpp/yaml.h>
#include "adf/include/log.h"

#define CONFIG_FILE_PATH "conf/soc_to_hmi.yaml"


int main(int argc, char* argv[]) {
    std::string binary_path = std::string(argv[0]);
    size_t pos = 0;
    for (size_t i = 0; i < binary_path.size(); i++) {
        if (binary_path[i] == '/') {
            pos = i;
        }
    }
    std::string folder_path = binary_path.substr(0, pos);
    std::string cm_conf_path = folder_path + "/../" + std::string(CONFIG_FILE_PATH);

    // 测试开关
    YAML::Node config = YAML::LoadFile(cm_conf_path);
    bool isTest = config["others"]["isTest"].as<bool>(false);
    hozon::netaos::extra::Server::Instance()->SetTest(isTest);
    std::cout << "isTest: " << isTest;

    // 模式切换，1：driving, 2：parking
    uint8_t running_mode = config["others"]["model"].as<uint8_t>(1);
    hozon::netaos::extra::Server::Instance()->SetMode(running_mode);
    std::cout << "Mode: " << running_mode;

    hozon::netaos::extra::Server::Instance()->Start(cm_conf_path);
    hozon::netaos::extra::Server::Instance()->NeedStopBlocking();
    hozon::netaos::extra::Server::Instance()->Stop();
    return 0;
}