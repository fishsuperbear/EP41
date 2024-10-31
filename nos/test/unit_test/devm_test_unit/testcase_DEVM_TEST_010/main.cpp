
#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <unordered_map>
#include "gtest/gtest.h"
#include "json/json.h"
#include "cfg_data.hpp"

using namespace hozon::netaos::unit_test;
TEST(DevmUpgradeTest, UpgradeVersion) {

    char buffer[1024]{};
    std::unordered_map<std::string, std::string> result_ver{};
    FILE *fp = popen("nos devm upgrade version", "r");
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        std::istringstream lineStream(buffer);
        std::string key{}, value{};
        lineStream >> key >> value;
        result_ver.insert({key, value});
        std::cout << "------------ " << buffer;
    }
    pclose(fp);

    const std::unordered_map<std::string, std::string> upgrade_version_tables = {
        {"major",               "dids/F1C0"         },
        {"soc",                 "version/SOC"       },
        {"mcu",                 "version/MCU"       },
        {"switch",              "version/SWT"       },
        {"SRR_FR",              "version/SRR_FR"    },
        {"SRR_RL",              "version/SRR_RL"    },
        {"LIDAR",               "version/LIDAR"     },
        {"SRR_FL",              "version/SRR_FL"    },
        {"SRR_RR",              "version/SRR_RR"    }
    };


    std::string sensor_ver{};
    for(const auto& pair : upgrade_version_tables) {
        sensor_ver.clear();
        if(pair.second.find("dids") != std::string::npos) {
            sensor_ver = CfgValueInfo::getInstance()->GetCfgValueFromFile("/cfg/dids/dids.json", pair.second);
        }
        else {
            sensor_ver = CfgValueInfo::getInstance()->GetCfgValueFromFile("/cfg/version/version.json", pair.second);
        }
        if((sensor_ver.size()>0?sensor_ver:"null") != result_ver[pair.first]) {
            std::cout << "UpgradeVersion" << pair.second << " " << sensor_ver << "," << result_ver[pair.first] << std::endl;
        }
        EXPECT_TRUE((sensor_ver.size()>0?sensor_ver:"null") == result_ver[pair.first]);
    }
}
TEST(DevmUpgradeTest, UpgradeStatus) {

    char buffer[1024]{};
    FILE *fp = popen("nos devm upgrade status", "r");
    while (fread(buffer, 1, sizeof(buffer), fp) > 0) {
        ;
    }
    pclose(fp);

    char *p = strstr(buffer, "status:");
    EXPECT_TRUE(p != NULL);
}
TEST(DevmUpgradeTest, UpgradePrecheck) {

    char buffer[1024]{};
    FILE *fp = popen("nos devm upgrade precheck", "r");
    while (fread(buffer, 1, sizeof(buffer), fp) > 0) {
        ;
    }
    pclose(fp);

    char *p = strstr(buffer, "precheck succ.");
    EXPECT_TRUE(p != NULL);
}
TEST(DevmUpgradeTest, UpgradeResult) {

//比对文件结果和命令结果
    std::unordered_map<std::string, std::string> result_res{};
    char buffer[1024]{};
    FILE *fp = popen("nos devm upgrade result", "r");
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        std::istringstream lineStream(buffer);
        std::string key{}, value{};
        lineStream >> key >> value;
        result_res.insert({key, value});
    }

    std::ifstream ifs("/opt/usr/log/ota_log/ota_result.json");
    ASSERT_TRUE(ifs.is_open());
    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    ASSERT_TRUE(res && errs.empty());
    ifs.close();

    const std::vector<std::string> res_tables{"ORIN", "LIDAR", "SRR_FL", "SRR_FR", "SRR_RL", "SRR_RR"};
    for (const auto& vec : res_tables) {
        if (root[vec].asString() != result_res[vec]) {
            std::cout << "key: " << vec << ",int_file " << root[vec].asString() << ",devm_get " << result_res[vec] << std::endl;
        }
        EXPECT_TRUE(root[vec].asString() == result_res[vec]);
    }
}
TEST(DevmUpgradeTest, UpgradePartition) {

    char buffer[1024]{};
    FILE *fp = popen("nos devm upgrade cur_partition", "r");
    while (fread(buffer, 1, sizeof(buffer), fp) > 0) {
        ;
    }
    pclose(fp);

    char *p = strstr(buffer, "current:");
    EXPECT_TRUE(p != NULL);
}
TEST(DevmUpgradeTest, UpgradeFinish) {

    char buffer[1024]{};
    FILE *fp = popen("nos devm upgrade finish", "r");
    while (fread(buffer, 1, sizeof(buffer), fp) > 0) {
        ;
    }
    pclose(fp);

    EXPECT_TRUE(strlen(buffer) > 0);
}
int32_t main(int32_t argc, char* argv[])
{
    printf("devm test\n");
    testing::InitGoogleTest(&argc,argv);
    int res = RUN_ALL_TESTS();
    return res;
}

