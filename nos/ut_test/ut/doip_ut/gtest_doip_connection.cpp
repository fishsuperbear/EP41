
#include "gtest/gtest.h"
#include "base/doip_connection.h"
TEST(DoipConnection, GetFd) {
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::DoipConnection *doipconnect = new hozon::netaos::diag::DoipConnection(fd);

    EXPECT_EQ(fd, doipconnect->GetFd());
    doipconnect->~DoipConnection();
    EXPECT_EQ(-1, doipconnect->GetFd());

    delete doipconnect;
}

TEST(DoipConnection, SetFd) {
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::DoipConnection *doipconnect = new hozon::netaos::diag::DoipConnection(fd);

    EXPECT_EQ(fd, doipconnect->GetFd());
    int32_t fd2 = socket(AF_INET, SOCK_STREAM, 0);
    doipconnect->SetFd(fd2);
    EXPECT_EQ(fd2, doipconnect->GetFd());

    delete doipconnect;
    close(fd);
}

