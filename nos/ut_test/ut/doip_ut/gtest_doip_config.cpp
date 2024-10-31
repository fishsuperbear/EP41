#include <iostream>
#include <memory>
#include <cstring>

#include "gtest/gtest.h"
#include "config/doip_config.h"


#define EXPECT_BINARY_EQ(expected, actual, size) \
    EXPECT_EQ(0, std::memcmp(expected, actual, size))
#define EXPECT_BINARY_NE(expected, actual, size) \
    EXPECT_NE(0, std::memcmp(expected, actual, size))

TEST(DoIPConfig, GetJsonAll) {
    bool ret = hozon::netaos::diag::DoIPConfig::Instance()->LoadConfig();
    ASSERT_EQ(true, ret);
}
TEST(DoIPConfig, SetVIN) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetVIN((char *)"0123456789abcdefg",DOIP_VIN_SIZE);
    char *vin = hozon::netaos::diag::DoIPConfig::Instance()->GetVIN();
    EXPECT_STREQ("0123456789abcdefg", vin);
    EXPECT_BINARY_EQ("0123456789abcdefg", vin, DOIP_VIN_SIZE);
}
TEST(DoIPConfig, SetVIN_nullptr) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetVIN(nullptr, 0);
    char *vin = hozon::netaos::diag::DoIPConfig::Instance()->GetVIN();
    EXPECT_STREQ("0123456789abcdefg", vin);
}
TEST(DoIPConfig, SetVIN_err_len) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetVIN((char *)"0123456789", DOIP_VIN_SIZE-1);
    char *vin = hozon::netaos::diag::DoIPConfig::Instance()->GetVIN();
    EXPECT_STRNE("0123456789", vin);
}


TEST(DoIPConfig, SetGID) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetGID((char *)"012345", DOIP_GID_SIZE);
    char *gid = hozon::netaos::diag::DoIPConfig::Instance()->GetGID();
    EXPECT_STREQ("012345", gid);
    EXPECT_BINARY_EQ("012345", gid, DOIP_GID_SIZE);
}
TEST(DoIPConfig, SetGID_nullptr) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetGID(nullptr, 0);
    char *gid = hozon::netaos::diag::DoIPConfig::Instance()->GetGID();
    EXPECT_STREQ("012345", gid);
    EXPECT_BINARY_EQ("012345", gid, DOIP_GID_SIZE);
}
TEST(DoIPConfig, SetGID_err_len) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetGID((char *)"012345", DOIP_GID_SIZE-1);
    char *gid = hozon::netaos::diag::DoIPConfig::Instance()->GetGID();
    EXPECT_STREQ("012345", gid);
    EXPECT_BINARY_EQ("012345", gid, DOIP_GID_SIZE);
}


TEST(DoIPConfig, SetEID) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetEID((char *)"012345", DOIP_EID_SIZE);
    char *gid = hozon::netaos::diag::DoIPConfig::Instance()->GetEID();
    EXPECT_STREQ("012345", gid);
    EXPECT_BINARY_EQ("012345", gid, DOIP_EID_SIZE);
}
TEST(DoIPConfig, SetEID_nullptr) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetEID(nullptr, 0);
    char *gid = hozon::netaos::diag::DoIPConfig::Instance()->GetEID();
    EXPECT_STREQ("012345", gid);
    EXPECT_BINARY_EQ("012345", gid, DOIP_EID_SIZE);
}
TEST(DoIPConfig, SetEID_err_len) {
    hozon::netaos::diag::DoIPConfig::Instance()->SetEID((char *)"012345", DOIP_EID_SIZE-1);
    char *gid = hozon::netaos::diag::DoIPConfig::Instance()->GetEID();
    EXPECT_STREQ("012345", gid);
    EXPECT_BINARY_EQ("012345", gid, DOIP_EID_SIZE);
}


TEST(DoIPConfig, LoadConfig_open_err) {
    int
    ret_sys = system("mv /app/runtime_service/diag_server/conf/doip_config.json /app/runtime_service/diag_server/conf/doip_config.jsonx");
    ASSERT_TRUE(0 == ret_sys);
    bool ret = hozon::netaos::diag::DoIPConfig::Instance()->LoadConfig();
    ret_sys = system("mv /app/runtime_service/diag_server/conf/doip_config.jsonx /app/runtime_service/diag_server/conf/doip_config.json");
    ASSERT_TRUE(0 == ret_sys);
    EXPECT_FALSE(ret);
}
TEST(DoIPConfig, LoadConfig_parse_err) {
    int
    ret_sys = system("mv /app/runtime_service/diag_server/conf/doip_config.json /app/runtime_service/diag_server/conf/doip_config.jsonx");
    ASSERT_TRUE(0 == ret_sys);
    ret_sys = system("echo 12345 /app/runtime_service/diag_server/conf/doip_config.json");
    ASSERT_TRUE(0 == ret_sys);
    bool ret = hozon::netaos::diag::DoIPConfig::Instance()->LoadConfig();
    ret_sys = system("mv /app/runtime_service/diag_server/conf/doip_config.jsonx /app/runtime_service/diag_server/conf/doip_config.json");
    ASSERT_TRUE(0 == ret_sys);
    EXPECT_FALSE(ret);
}


TEST(DoIPConfig, GetRoutingIp) {
    char *ip = hozon::netaos::diag::DoIPConfig::Instance()->GetRoutingIp(4298);
    EXPECT_TRUE(strlen(ip) != 0);
    ip = hozon::netaos::diag::DoIPConfig::Instance()->GetRoutingIp(0x10c4);
    EXPECT_TRUE(strlen(ip) != 0);
    std::cout << "ip " << ip << std::endl;
}

TEST(DoIPConfig, GetIfNameByType_client) {
    std::string ifname = hozon::netaos::diag::DoIPConfig::Instance()->GetIfNameByType(hozon::netaos::diag::DOIP_IF_USE_DOIP_CLIENT);
    EXPECT_TRUE(ifname.size() != 0);
    std::cout << "ifname " << ifname << std::endl;
}
TEST(DoIPConfig, GetIfNameByType_server) {
    std::string ifname = hozon::netaos::diag::DoIPConfig::Instance()->GetIfNameByType(hozon::netaos::diag::DOIP_IF_USE_DOIP_SERVER);
    EXPECT_TRUE(ifname.size() != 0);
    std::cout << "ifname " << ifname << std::endl;
}
TEST(DoIPConfig, GetIfNameByType_both) {
    std::string ifname = hozon::netaos::diag::DoIPConfig::Instance()->GetIfNameByType(hozon::netaos::diag::DOIP_IF_USE_DOIP_BOTH);
    EXPECT_TRUE(ifname.size() == 0);
    std::cout << "ifname " << ifname << std::endl;
}
TEST(DoIPConfig, GetIfNameByType_nullptr) {
    std::string ifname = hozon::netaos::diag::DoIPConfig::Instance()->GetIfNameByType((hozon::netaos::diag::doip_if_use_t)0xff);
    EXPECT_TRUE(ifname.size() == 0);
    std::cout << "ifname " << ifname << std::endl;
}


TEST(DoIPConfig, GetIfNameByIp) {
    std::string ifname = hozon::netaos::diag::DoIPConfig::Instance()->GetIfNameByIp((char *)"nullptr");
    EXPECT_TRUE(ifname.size() == 0);
    std::cout << "ifname " << ifname << std::endl;
}
TEST(DoIPConfig, GetIfNameByIp_ip) {
    std::string ifname = hozon::netaos::diag::DoIPConfig::Instance()->GetIfNameByIp((char *)"10.4.51.71");
    EXPECT_TRUE(ifname.size() != 0);
    std::cout << "ifname " << ifname << std::endl;
}


TEST(DoIPConfig, GetNetSourceList) {
    std::vector<hozon::netaos::diag::doip_net_source_t*> ifname = hozon::netaos::diag::DoIPConfig::Instance()->GetNetSourceList();
    EXPECT_TRUE(ifname.size() != 0);
}

TEST(DoIPConfig, GetMaxRequestBytes) {
    uint32_t bytes = hozon::netaos::diag::DoIPConfig::Instance()->GetMaxRequestBytes(1);
    EXPECT_TRUE(bytes != 0);
    std::cout << "bytes " << bytes << std::endl;
}


