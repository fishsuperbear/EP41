


#include "gtest/gtest.h"
#include "base/doip_select.h"
#include <sys/socket.h>
#include <fcntl.h>
TEST(DoipSelect, Create) {
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}

// TEST(DoipSelect, Destroy) {
//     int32_t ret;
//     hozon::netaos::diag::doip_event_t ep;
//     memset(&ep, 0, sizeof ep);
//     int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
//     // if (mask & hozon::netaos::diag::DOIP_EVENT_READABLE) {
//     //     ep.events |= hozon::netaos::diag::DOIP_EV_READ;
//     // }
//     // if (mask & hozon::netaos::diag::DOIP_EVENT_WRITABLE) {
//     //     ep.events |= hozon::netaos::diag::DOIP_EV_WRITE;
//     // }
//     // ep.data.ptr = source;
//     ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_ADD, fd, &ep);
//     EXPECT_EQ(ret, 0);
//     ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_DEL, fd, &ep);
//     EXPECT_EQ(ret, 0);
//     close(fd);
// }

TEST(DoipSelect, Control) {
    int32_t ret;
    hozon::netaos::diag::doip_event_t ep;
    memset(&ep, 0, sizeof ep);
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);

    hozon::netaos::diag::DoipSelect::Instance()->Create();
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_ADD, fd, &ep);
    EXPECT_EQ(ret, 0);
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_ADD, fd, &ep);
    EXPECT_EQ(ret, -1);
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_ADD, -1, &ep);
    EXPECT_EQ(ret, -1);

    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_MOD, fd, &ep);
    EXPECT_EQ(ret, 0);
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_MOD, -1, &ep);
    EXPECT_EQ(ret, -1);
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_MOD, 0xFFFF, &ep);
    EXPECT_EQ(ret, -1);

    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_DEL, fd, &ep);
    EXPECT_EQ(ret, 0);
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_DEL, -1, &ep);
    EXPECT_EQ(ret, -1);
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_DEL, 0xFFFF, &ep);
    EXPECT_EQ(ret, -1);
    
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_DEL+100, fd, &ep);
    EXPECT_EQ(ret, -1);
    close(fd);
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}

TEST(DoipSelect, Dispatch) {
#define ARRAY_LENGTH(a) (sizeof (a) / sizeof (a)[0])
    int32_t ret;
    int32_t timeout = 1000;//ms
    hozon::netaos::diag::doip_event_t ep[10];
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_TRUE(fd >= 0);
    
    // struct sockaddr_in addr;
    // addr.sin_family = AF_INET;
    // addr.sin_port = htons(13402);
    // addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    // ret = connect(fd, (struct sockaddr *)&addr, sizeof addr);
    // ASSERT_TRUE(ret == 0);
    // printf("==== fd %d\n", fd);
    
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    hozon::netaos::diag::doip_event_t ea;
    memset(&ea, 0, sizeof ea);
    
    ret = hozon::netaos::diag::DoipSelect::Instance()->Control(DOIP_SEL_ADD, fd, &ea);
    EXPECT_EQ(ret, 0);

    int32_t count = hozon::netaos::diag::DoipSelect::Instance()->Dispatch(ep, ARRAY_LENGTH(ep), timeout);
    ASSERT_TRUE(count >= 0);
    
    close(fd);
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}

TEST(DoipSelect, Notify) {
    
}
