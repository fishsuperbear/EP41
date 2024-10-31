

#include "gtest/gtest.h"
#include "socket/doip_socket_os.h"
TEST(DoipSocketOS, SetCloexecOrClose) {

}

TEST(DoipSocketOS, CreateSocket) {
    int32_t
    fd = hozon::netaos::diag::DoipSocketOS::CreateSocket(AF_INET, SOCK_STREAM, 0);
    EXPECT_TRUE(fd > 0);
    close(fd);
    fd = hozon::netaos::diag::DoipSocketOS::CreateSocket(0, SOCK_STREAM, 0);
    EXPECT_EQ(fd, -1);
}

TEST(DoipSocketOS, Accept) {
    int32_t
    fd = hozon::netaos::diag::DoipSocketOS::CreateSocket(AF_INET, SOCK_STREAM, 0);
    EXPECT_TRUE(fd > 0);

    struct sockaddr_in addr;
    socklen_t length = sizeof addr;
    memset(&addr, 0, length);
    int32_t
    client_fd = hozon::netaos::diag::DoipSocketOS::Accept(fd, (struct sockaddr *) &addr, &length);
    //todo:需要建立服务端的socket，等待客户端连接
    printf("==============%d\n", client_fd);
    // EXPECT_TRUE(client_fd >= 0);
    
    close(fd);
}

TEST(DoipSocketOS, Connect) {

}
