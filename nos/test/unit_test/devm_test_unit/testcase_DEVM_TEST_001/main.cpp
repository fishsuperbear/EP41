
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <filesystem>
#include "gtest/gtest.h"

#include "cfg/include/config_param.h"
#include "get_ifdata.h"
#include "device_info.h"
#include "json/json.h"

using namespace hozon::netaos::cfg;
using namespace hozon::netaos::devm_server;
using namespace hozon::netaos::tools;

TEST(DevmFunctionTest, ReadDidInfo) {

    std::ifstream ifs("/cfg/dids/dids.json");
    ASSERT_TRUE(ifs.is_open());
    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    ASSERT_TRUE(res && errs.empty());
    ifs.close();

    Json::Value kv = root["kv_vec"];

    //***使用接口获取value
    // DevmClientDidInfo readdid;
    // for (const auto& key_value : kv) {
    //     std::string dids = key_value["key"].asString();
    //     std::string value_json = key_value["value"]["string"].asString();
    //     std::string value_read = readdid.ReadDidInfo(dids);

    //     if (value_json.size() > 0 && value_read.size() <= 0) {
    //         std::cout << "ReadDidInfo" << dids << "[" << value_json << "][" << value_read << "]" << std::endl;
    //     }
    //     if (value_json.size() > 0) {
    //         EXPECT_TRUE(value_read.size() > 0);
    //     }
    // }

    for (const auto& key_value : kv) {
        std::string dids = key_value["key"].asString();
        std::string value_json = key_value["value"]["string"].asString();
        for (size_t i = 0; i < value_json.length(); i++) {
            value_json[i] = std::toupper(value_json[i]);
        }

        std::string value_read{};
        char buffer[1024]{};
        std::string cmd("nos devm read-did 0x");
        cmd = cmd + dids;
        FILE *fp = popen(cmd.c_str(), "r");
        while (fread(buffer, 1, sizeof(buffer), fp) > 0)
        {
            value_read += buffer;
            memset(buffer, 0, sizeof(buffer));
        }
        pclose(fp);
        size_t colonPos = value_read.find(value_json);
        if (colonPos == std::string::npos) {
            std::cout << "ReadDidInfo " << dids << "[" << value_read << "][" << value_json << "]" << std::endl;
        }
        EXPECT_TRUE (colonPos != std::string::npos);

    }
}

TEST(DevmFunctionTest, device_info) {

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
    std::cout << "soc_ver " << "[" << result_ver["SOC"] << "][" << (soc_ver.size()>0?soc_ver:"null") << "]" << std::endl;
    EXPECT_TRUE((result_ver["SOC"] == (soc_ver.size()>0?soc_ver:"null")));


    EXPECT_TRUE((result_ver["MCU"] != "null"));
    EXPECT_TRUE((result_ver["SWT"] != "null"));
    std::ifstream file("/etc/version");
    EXPECT_TRUE(file.is_open());
    std::string dsv_version;
    std::getline(file, dsv_version);
    std::cout << "dsv_version " << "[" << result_ver["DSV"] << "][" << dsv_version << "]" << std::endl;
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

TEST(DevmFunctionTest, device_status) {

    const char *command = "nos devm dev-status";
    char buffer[1024]{};

    std::unordered_map<std::string, std::string> result{};
    FILE *fp = popen(command, "r");
    ASSERT_TRUE((fp != NULL));
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        std::string key(buffer, 20), value(buffer+20, 20);
        size_t pos = key.find_last_not_of(" ");
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

TEST(DevmFunctionTest, if_data_fail) {

    std::vector<std::string> arguments ={"si"};
    IfDataInfo get_ifdata(arguments);
    int32_t ret = get_ifdata.StartGetIfdata();
    EXPECT_TRUE(ret == -1);
}

// TEST(DevmFunctionTest, if_data) {
//     std::vector<std::string> arguments ={"si", "lo"};
//     IfDataInfo get_ifdata(arguments);
//     int32_t ret = get_ifdata.StartGetIfdata();
//     EXPECT_TRUE(ret == -1);
// }
TEST(DevmFunctionTest, cpu_info) {

    std::unordered_map<std::string, std::string> res_cpu_info{};
    char buffer[1024]{};
    FILE *fp = popen("nos devm cpu-info", "r");
    ASSERT_TRUE((fp != NULL));
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        std::istringstream lineStream(buffer);
        std::string key{}, value{}, colon{};
        lineStream >> key;
        value = buffer + 24 + 2;
        size_t pos = value.find_last_not_of(" \r\n");
        if (pos != std::string::npos) {
            value.erase(pos + 1);
        }
        value += " ";
        res_cpu_info.insert({key, value});
        std::cout << "------------ " << buffer;
    }
    pclose(fp);
#if defined(BUILD_FOR_ORIN)
    EXPECT_TRUE(res_cpu_info["Architecture:"] == "aarch64");
#endif
    std::unordered_map<std::string, std::string> process_name;
    for (const auto& entry : std::filesystem::recursive_directory_iterator("/app/runtime_service/")) {
        if (entry.is_regular_file() && entry.path().filename() == "MANIFEST.json") {
            std::string parent_path = entry.path().parent_path().parent_path().string();
            size_t pos = parent_path.find_last_of('/');
            if (pos != std::string::npos) {
                process_name[parent_path.substr(pos + 1)] = entry.path().string();
            }
        }
    }
    std::unordered_map<std::string, std::string> process_bind;
    for (auto pair: process_name) {
        std::ifstream ifs(pair.second);
        ASSERT_TRUE((ifs.is_open()));
        Json::CharReaderBuilder reader;
        Json::Value root;
        JSONCPP_STRING errs;
        bool res = Json::parseFromStream(reader, ifs, &root, &errs);
        ASSERT_TRUE(res && errs.empty());
        ifs.close();
        std::stringstream ss;
        for (auto vec : root["shall_run_ons"]) {
            ss << vec << " ";
        }
        process_bind[pair.first] = ss.str();
    }
    for (auto pair: process_bind) {
        if (res_cpu_info[pair.first] != pair.second) {
            std::cout << "cpu-info err, [" << res_cpu_info[pair.first] << "][" << pair.second << "]" << std::endl;
        }
        EXPECT_TRUE(res_cpu_info[pair.first] == pair.second);
    }
}

TEST(DevmFunctionTest, iostat) {

    char buffer[1024]{};
    FILE *fp = popen("nos devm iostat", "r");
    ASSERT_TRUE((fp != NULL));
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        ;//printf("%s", buffer);
    }
    pclose(fp);
    EXPECT_TRUE(strlen(buffer) > 0);
}

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
            std::cout << "UpgradeVersion " << pair.second << " " << sensor_ver << "," << result_ver[pair.first] << std::endl;
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
#if defined(BUILD_FOR_ORIN)
    EXPECT_TRUE(p != NULL);
#endif
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
#if defined(BUILD_FOR_ORIN)
    EXPECT_TRUE(p != NULL);
#endif
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

    ConfigParam::Instance()->Init(2000);
    int res = RUN_ALL_TESTS();
    ConfigParam::Instance()->DeInit();

    return res;
}

