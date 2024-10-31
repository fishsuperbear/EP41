
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <unordered_map>
#include "gtest/gtest.h"

#include "cfg/include/config_param.h"
#include "json/json.h"
#include "cfg_data.hpp"
#include "device_info.h"

using namespace hozon::netaos::cfg;
using namespace hozon::netaos::devm_server;

TEST(DevmDeviceInfo, device_status2) {

    const char *command = "nos devm dev-status";
    char buffer[1024]{};

    std::unordered_map<std::string, std::string> result{};
    FILE *fp = popen(command, "r");
    ASSERT_TRUE((fp != NULL));
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        std::string key(buffer, 20), value(buffer+20, 20);
        int pos = key.find_last_not_of(" ");
        if (pos != std::string::npos) {
            key.erase(pos + 1);
        }
        pos = value.find_last_not_of(" ");
        if (pos != std::string::npos) {
            value.erase(pos + 1);
        }
        result.insert({key, value});
        std::cout << "------------ " << buffer;
        memset(buffer, 0, sizeof(buffer));
        //printf("%s", buffer);
    }
    pclose(fp);

    std::ifstream ifs("/cfg/system/system.json");
    ASSERT_TRUE(ifs.is_open());
    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool resx = Json::parseFromStream(reader, ifs, &root, &errs);
    ASSERT_TRUE(resx && errs.empty());
    ifs.close();

    Json::Value kv = root["kv_vec"];
    for (const auto& key_value : kv) {
        std::string dids = key_value["key"].asString();
        std::string value_json = key_value["value"]["string"].asString();
        if (dids == "soc_status") {
            EXPECT_TRUE(result["soc"] == value_json);
        }
        if (dids == "muc_status") {
            EXPECT_TRUE(result["mcu"] == value_json);
        }
    }

    uint8_t status{};
    std::string str_status{};
    CfgResultCode res{};
    for(const auto& pair : camera_status_tables) {
        // 0x0-Unkown, 0x1-link locked, 0x2-link unlock
        status = 0;
        res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
        str_status = res != CONFIG_OK ? "Unkown"
                    : status == 0 ? "Unkown"
                    : status == 1 ? "Link_locked"
                    : "Link_unlock";
        if (str_status != result[pair.second]) {
            std::cout << "dev-status err, " << "[" << result[pair.second] << "][" << str_status << "] key " << pair.second << std::endl;
        }
        if (result[pair.second].size() > 0) {
            EXPECT_TRUE(str_status == result[pair.second]);
        }
    }
    for(const auto& pair : lidar_status_tables) {
        // 0x0-Unkown, 0x1-Working, 0x2-link unlock
        status = 0;
        res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
        str_status = res != CONFIG_OK ? "Unkown"
                    : status == 0 ? "Unkown"
                    : status == 1 ? "Working"
                    : "Not Working";
        if (str_status != result[pair.second]) {
            std::cout << "dev-status err, " << "[" << result[pair.second] << "][" << str_status << "] key " << pair.second << std::endl;
        }
        if (result[pair.second].size() > 0) {
            EXPECT_TRUE(str_status == result[pair.second]);
        }
    }
    for(const auto& pair : radar_status_tables) {
        status = 0;
        res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
        str_status = res != CONFIG_OK ? "Unkown"
                    : status == 0 ? "Unkown"
                    : status == 1 ? "Working"
                    : "Not Working";
        if (str_status != result[pair.second]) {
            std::cout << "dev-status err, " << "[" << result[pair.second] << "][" << str_status << "] key " << pair.second << std::endl;
        }
        if (result[pair.second].size() > 0) {
            EXPECT_TRUE(str_status == result[pair.second]);
        }
    }
    for(const auto& pair : uss_status_tables) {
        status = 0;
        res = ConfigParam::Instance()->GetParam<uint8_t>(pair.first, status);
        str_status = res != CONFIG_OK ? "Unkown"
                    : status == 0 ? "Unkown"
                    : status == 1 ? "Working"
                    : "Not Working";
        if (str_status != result[pair.second]) {
            std::cout << "dev-status err, " << "[" << result[pair.second] << "][" << str_status << "] key " << pair.second << std::endl;
        }
        if (result[pair.second].size() > 0) {
            EXPECT_TRUE(str_status == result[pair.second]);
        }
    }
}

int32_t main(int32_t argc, char* argv[])
{
    printf("devm test\n");
    testing::InitGoogleTest(&argc,argv);

    ConfigParam::Instance()->Init(2000);
    int res = RUN_ALL_TESTS();
    ConfigParam::Instance()->DeInit();

    return res;
}

