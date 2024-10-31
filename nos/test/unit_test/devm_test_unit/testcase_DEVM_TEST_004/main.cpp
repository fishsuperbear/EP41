
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

TEST(DevmDeviceInfo, device_info4) {

    int dev_type = 0;
    char buffer[1024]{};
    std::unordered_map<std::string, std::string> result_ver{};
    std::unordered_map<std::string, std::string> result_type{};
    std::unordered_map<std::string, std::string> result_temp{};
    std::unordered_map<std::string, std::string> result_vol{};
    FILE *fp = popen("nos devm dev-info", "r");
    ASSERT_TRUE((fp != NULL));
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        if (strstr(buffer, "Device") != NULL) {
            dev_type++;
            continue;
        }
        if (dev_type == 1) {
            std::istringstream lineStream(buffer);
            std::string key{}, value{};
            lineStream >> key >> value;
            result_ver.insert({key, value});
        }
        if (dev_type == 2) {
            std::istringstream lineStream(buffer);
            std::string key{}, value{};
            lineStream >> key >> value;
            result_type.insert({key, value});
        }
        if (dev_type == 3) {
            std::istringstream lineStream(buffer);
            std::string key{}, value{};
            lineStream >> key >> value;
            result_temp.insert({key, value});
        }
        if (dev_type == 4) {
            std::istringstream lineStream(buffer);
            std::string key{}, value{};
            lineStream >> key >> value;
            result_vol.insert({key, value});
        }
        std::cout << "------------ " << buffer;
        memset(buffer, 0, sizeof(buffer));
    }
    pclose(fp);

    std::ifstream ifs("/app/version.json");
    ASSERT_TRUE(ifs.is_open());
    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool resx = Json::parseFromStream(reader, ifs, &root, &errs);
    ASSERT_TRUE(resx && errs.empty());
    ifs.close();
    std::string soc_ver = root["app_version"].asString();
    if (result_ver["SOC"] != (soc_ver.size()>0?soc_ver:"null")) {
        std::cout << "soc_ver " << "[" << result_ver["SOC"] << "][" << (soc_ver.size()>0?soc_ver:"null") << "]" << std::endl;
    }
    EXPECT_TRUE((result_ver["SOC"] == (soc_ver.size()>0?soc_ver:"null")));


    EXPECT_TRUE((result_ver["MCU"] != "null"));
    EXPECT_TRUE((result_ver["SWT"] != "null"));
    std::ifstream file("/etc/version");
    EXPECT_TRUE(file.is_open());
    std::string dsv_version;
    std::getline(file, dsv_version);
    if (dsv_version != result_ver["DSV"]) {
        std::cout << "dsv_version " << "[" << result_ver["DSV"] << "][" << dsv_version << "]" << std::endl;
    }
    EXPECT_TRUE((dsv_version == result_ver["DSV"]));

    std::string sensor_ver{};
    for(const auto& pair : version_tables) {
        sensor_ver.clear();
        auto res = ConfigParam::Instance()->GetParam<std::string>(pair.first, sensor_ver);
        if ((res == CONFIG_OK)) {
            EXPECT_TRUE((sensor_ver.size()>0?sensor_ver:"null") == result_ver[pair.second]);
        }
    }


    EXPECT_TRUE(result_type["soc"] == "OrinX");
    EXPECT_TRUE(result_type["mcu"] == "TC397");
    EXPECT_TRUE(result_type["swt"] == "Marvell");


    // EXPECT_TRUE(result_type["temp_soc"] == "1111");
    // EXPECT_TRUE(result_type["temp_mcu"] == "1111");
    // EXPECT_TRUE(result_type["temp_ext0"] == "1111");
    // EXPECT_TRUE(result_type["temp_ext1"] == "1111");
    
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

