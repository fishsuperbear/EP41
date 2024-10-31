

#include "gtest/gtest.h"
#include "base/doip_util.h"

TEST(DoipUtil, DoipBswap16) {
    uint16_t  swap16;
    swap16 = hozon::netaos::diag::DoipUtil::Instance().DoipBswap16(1);
    EXPECT_EQ(swap16, 0x100);
    swap16 = hozon::netaos::diag::DoipUtil::Instance().DoipBswap16(0x12);
    EXPECT_EQ(swap16, 0x1200);
    swap16 = hozon::netaos::diag::DoipUtil::Instance().DoipBswap16(0xff);
    EXPECT_EQ(swap16, 0xff00);
    swap16 = hozon::netaos::diag::DoipUtil::Instance().DoipBswap16(0);
    EXPECT_EQ(swap16, 0);
    swap16 = hozon::netaos::diag::DoipUtil::Instance().DoipBswap16(0xffff);
    EXPECT_EQ(swap16, 0xffff);
}
TEST(DoipUtil, DoipGetRandomValue) {
    int32_t random, random2;
    random = hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1);
    EXPECT_EQ(random, 1);
    random = hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(2);
    EXPECT_TRUE(random >= 1 && random <=2);
    random = hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024);
    random2 = hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024);
    EXPECT_TRUE(random >= 1 && random <=1024);
    EXPECT_TRUE(random2 >= 1 && random2 <=1024);
    EXPECT_TRUE(random != random2);

    // std::cout << "xxxxxxxxxxx: " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << " " << hozon::netaos::diag::DoipUtil::Instance().DoipGetRandomValue(1024)
    //     << std::endl;
}

TEST(DoipUtil, IsInternalEquipmentAddress) {
    EXPECT_TRUE(hozon::netaos::diag::DoipUtil::Instance().IsInternalEquipmentAddress(0x10c3));
}

TEST(DoipUtil, IsFunctianalAddress_true) {
    EXPECT_TRUE(hozon::netaos::diag::DoipUtil::Instance().IsFunctianalAddress(58368));
}
TEST(DoipUtil, IsFunctianalAddress_false) {
    EXPECT_FALSE(hozon::netaos::diag::DoipUtil::Instance().IsFunctianalAddress(0xffff));
}

TEST(DoipUtil, IsTestEquipmentAddress_true) {
    EXPECT_TRUE(hozon::netaos::diag::DoipUtil::Instance().IsTestEquipmentAddress(4195));
}
TEST(DoipUtil, IsTestEquipmentAddress_false) {
    EXPECT_FALSE(hozon::netaos::diag::DoipUtil::Instance().IsTestEquipmentAddress(0xffff));
}

TEST(DoipUtil, IsInternalEquipmentAddress_true) {
    EXPECT_TRUE(hozon::netaos::diag::DoipUtil::Instance().IsInternalEquipmentAddress(1));
}

TEST(DoipUtil, DoipNodeUdpListFindByIpPort) {
    std::list<hozon::netaos::diag::doip_node_udp_table_t*> node_list;
    hozon::netaos::diag::doip_node_udp_table_t node={"1.1.1.1", 1234};
    node_list.push_back(&node);
    EXPECT_TRUE(hozon::netaos::diag::DoipUtil::Instance().DoipNodeUdpListFindByIpPort(node_list, (char *)"1.1.1.1", 1234));
    EXPECT_FALSE(hozon::netaos::diag::DoipUtil::Instance().DoipNodeUdpListFindByIpPort(node_list, (char *)"1.1.1.1", 1111));
    EXPECT_FALSE(hozon::netaos::diag::DoipUtil::Instance().DoipNodeUdpListFindByIpPort(node_list, (char *)"1.1.1.2", 0));
}

TEST(DoipUtil, DoipNodeTcpListFindByFd) {
    std::list<hozon::netaos::diag::doip_node_tcp_table_t*> node_list;
    hozon::netaos::diag::doip_node_tcp_table_t node={"1.1.1.1", 1234, 10};
    node_list.push_back(&node);
    EXPECT_TRUE(nullptr != hozon::netaos::diag::DoipUtil::Instance().DoipNodeTcpListFindByFd(node_list,10));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipNodeTcpListFindByFd(node_list,1));
}

TEST(DoipUtil, DoipNodeTcpListFindByLA) {
    std::list<hozon::netaos::diag::doip_node_tcp_table_t*> node_list;
    hozon::netaos::diag::doip_node_tcp_table_t node={"1.1.1.1", 1234, 10, 0x1234};
    node_list.push_back(&node);
    EXPECT_TRUE(nullptr != hozon::netaos::diag::DoipUtil::Instance().DoipNodeTcpListFindByLA(node_list,0x1234));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipNodeTcpListFindByLA(node_list,1));
}
TEST(DoipUtil, DoipNodeTcpListRegistFdCount) {
    std::list<hozon::netaos::diag::doip_node_tcp_table_t*> node_list;
    hozon::netaos::diag::doip_node_tcp_table_t node={"1.1.1.1", 1, 2, 3, 4, 5, 6, 7, 8, hozon::netaos::diag::DOIP_CONNECT_STATE_REGISTERED_ROUTING_ACTIVE};
    node_list.push_back(&node);
    EXPECT_EQ(1, hozon::netaos::diag::DoipUtil::Instance().DoipNodeTcpListRegistFdCount(node_list));
}

TEST(DoipUtil, DoipEquipUdpListFindByEID) {
    std::list<hozon::netaos::diag::doip_equip_udp_table_t*> node_list;
    uint64_t eid = 1234;
    hozon::netaos::diag::doip_equip_udp_table_t node={1, "1.1.1.1", 1234, 0x10c4};
    memcpy(node.eid, &eid, DOIP_EID_SIZE);
    node_list.push_back(&node);
    EXPECT_TRUE(nullptr != hozon::netaos::diag::DoipUtil::Instance().DoipEquipUdpListFindByEID(node_list, (char *)&eid));
    eid = 123456;
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipUdpListFindByEID(node_list, (char *)&eid));
}
TEST(DoipUtil, DoipEquipUdpListFindByIpPort) {
    std::list<hozon::netaos::diag::doip_equip_udp_table_t*> node_list;
    hozon::netaos::diag::doip_equip_udp_table_t node={1, "1.1.1.1", 1234, 0x10c4};
    node_list.push_back(&node);
    EXPECT_TRUE(nullptr != hozon::netaos::diag::DoipUtil::Instance().DoipEquipUdpListFindByIpPort(node_list, (char *)"1.1.1.1", 1234));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipUdpListFindByIpPort(node_list, (char *)"1.1.1.1", 1));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipUdpListFindByIpPort(node_list, (char *)"000", 1234));
}


TEST(DoipUtil, DoipEquipTcpListFindByFd) {
    std::list<hozon::netaos::diag::doip_equip_tcp_table_t*> node_list;
    hozon::netaos::diag::doip_equip_tcp_table_t node={10, "1.1.1.1", 1234, 0x10c4, 0x1042};
    node_list.push_back(&node);
    EXPECT_TRUE(nullptr != hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByFd(node_list, 10));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByFd(node_list, 0));
}
TEST(DoipUtil, DoipEquipTcpListFindByLA) {
    std::list<hozon::netaos::diag::doip_equip_tcp_table_t*> node_list;
    hozon::netaos::diag::doip_equip_tcp_table_t node={10, "1.1.1.1", 1234, 0x10c4, 0x1042};
    node_list.push_back(&node);
    EXPECT_TRUE(nullptr != hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByLA(node_list, 0x1042, 0x10c4));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByLA(node_list, 0x1042, 1));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByLA(node_list, 0, 0));
}
TEST(DoipUtil, DoipEquipTcpListFindByIPandLA) {
    std::list<hozon::netaos::diag::doip_equip_tcp_table_t*> node_list;
    hozon::netaos::diag::doip_equip_tcp_table_t node={10, "1.1.1.1", 1234, 0x10c4, 0x1042};
    node_list.push_back(&node);
    EXPECT_TRUE(nullptr != hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByIPandLA(node_list, (char *)"1.1.1.1", 0x1042));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByIPandLA(node_list, (char *)"1.1.1.1", 1));
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipUtil::Instance().DoipEquipTcpListFindByIPandLA(node_list, (char *)"000", 0x1042));
}

