

#include "gtest/gtest.h"
#include "base/doip_netlink.h"

TEST(DoipNetlink, GetIFName) {
    std::unique_ptr<hozon::netaos::diag::DoipNetlink> net_link_;
    net_link_ = std::make_unique<hozon::netaos::diag::DoipNetlink>();
    char fname[100] = {0};
    
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_TRUE(fd >= 0);

    net_link_->GetIFName(fname, fd, nullptr);
    EXPECT_STREQ("", fname);
    net_link_->GetIFName(nullptr, fd, (char *)"127.0.0.1");
    EXPECT_STREQ("", fname);
    
    net_link_->GetIFName(fname, 0xffff, (char *)"127.0.0.1");
    EXPECT_STREQ("", fname);
    
    net_link_->GetIFName(fname, fd, (char *)"127.0.0.1");
    EXPECT_STREQ("lo", fname);
    close(fd);
}

TEST(DoipNetlink, GetIp) {
    std::unique_ptr<hozon::netaos::diag::DoipNetlink> net_link_;
    net_link_ = std::make_unique<hozon::netaos::diag::DoipNetlink>();
    char ip[100];

    memset(ip, 0, sizeof ip);
    int32_t ret = net_link_->GetIp((char *)"lo", ip, sizeof ip);
    EXPECT_EQ(0, ret);
    EXPECT_STREQ("127.0.0.1", ip);
    
    memset(ip, 0, sizeof ip);
    ret = net_link_->GetIp(nullptr, ip, sizeof ip);
    EXPECT_EQ(-1, ret);
    
    memset(ip, 0, sizeof ip);
    ret = net_link_->GetIp((char *)"lo", nullptr, 0);
    EXPECT_EQ(-1, ret);
    
    memset(ip, 0, sizeof ip);
    ret = net_link_->GetIp((char *)"lo", ip, 0);
    EXPECT_EQ(0, ret);
    EXPECT_STREQ("", ip);
    
}

TEST(DoipNetlink, GetMac) {
    std::unique_ptr<hozon::netaos::diag::DoipNetlink> net_link_;
    net_link_ = std::make_unique<hozon::netaos::diag::DoipNetlink>();
    char mac[100];
    
    memset(mac, 0, sizeof mac);
    int32_t ret = net_link_->GetMac((char *)"lo", mac);
    EXPECT_EQ(0, ret);
}

TEST(DoipNetlink, CheckLinkAvailable) {
    
    std::unique_ptr<hozon::netaos::diag::DoipNetlink> net_link_;
    net_link_ = std::make_unique<hozon::netaos::diag::DoipNetlink>();

    std::string ifname("lo");
    int32_t ret = net_link_->CheckLinkAvailable(ifname, nullptr);
    EXPECT_EQ(11, ret);
    
    ifname = "123";
    ret = net_link_->CheckLinkAvailable(ifname, nullptr);
    EXPECT_EQ(0, ret);
    
    int
    ret_sys = system("sudo ifconfig lo down");
    ASSERT_TRUE(0 == ret_sys);
    ifname = "lo";
    ret = net_link_->CheckLinkAvailable(ifname, nullptr);
    EXPECT_EQ(1, ret);
    ret_sys = system("sudo ifconfig lo up");
    ASSERT_TRUE(0 == ret_sys);
    
    ifname = "lo";
    ret = net_link_->CheckLinkAvailable(ifname, nullptr);
    EXPECT_EQ(11, ret);
}

