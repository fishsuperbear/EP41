

#include <sys/socket.h>
#include <fcntl.h>
#include "gtest/gtest.h"
#include "base/doip_event_loop.h"

int32_t DoipEventLoop_test_call_back(int32_t fd, uint32_t mask, void *) {
    printf("DoipEventLoop_test_call_back call...\n");
    return 0;
}
TEST(DoipEventLoop, SourceCreate) {
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    std::shared_ptr<hozon::netaos::diag::DoipEventLoop> event_loop_;
    event_loop_ = std::make_shared<hozon::netaos::diag::DoipEventLoop>();

    uint8_t *data = new uint8_t[4];
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::doip_event_source_t* source = event_loop_->SourceCreate(fd, DoipEventLoop_test_call_back, data);
    EXPECT_NE(nullptr, source);
    event_loop_->SourceDestroy(source);
    delete[] data;
    close(fd);
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}
TEST(DoipEventLoop, SourceDestroy) {
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    std::shared_ptr<hozon::netaos::diag::DoipEventLoop> event_loop_;
    event_loop_ = std::make_shared<hozon::netaos::diag::DoipEventLoop>();

    uint8_t *data = new uint8_t[4];
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::doip_event_source_t* source = event_loop_->SourceCreate(fd, DoipEventLoop_test_call_back, data);
    EXPECT_NE(nullptr, source);
    event_loop_->SourceDestroy(source);
    delete[] data;
    close(fd);
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}
TEST(DoipEventLoop, SourceAdd) {
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    std::shared_ptr<hozon::netaos::diag::DoipEventLoop> event_loop_;
    event_loop_ = std::make_shared<hozon::netaos::diag::DoipEventLoop>();

    uint8_t *data = new uint8_t[4];
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::doip_event_source_t* source = event_loop_->SourceCreate(fd, DoipEventLoop_test_call_back, data);
    ASSERT_NE(nullptr, source);
    
    int32_t 
    ret = event_loop_->SourceAdd(source, hozon::netaos::diag::DOIP_EVENT_READABLE);
    EXPECT_EQ(0, ret);
    ret = event_loop_->SourceAdd(source, hozon::netaos::diag::DOIP_EVENT_READABLE);
    EXPECT_EQ(-1, ret);

    ret = event_loop_->SourceRemove(source, 0);
    EXPECT_EQ(0, ret);
    event_loop_->SourceDestroy(source);
    delete[] data;
    close(fd);
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}
TEST(DoipEventLoop, SourceRemove) {
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    std::shared_ptr<hozon::netaos::diag::DoipEventLoop> event_loop_;
    event_loop_ = std::make_shared<hozon::netaos::diag::DoipEventLoop>();

    uint8_t *data = new uint8_t[4];
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::doip_event_source_t* source = event_loop_->SourceCreate(fd, DoipEventLoop_test_call_back, data);
    ASSERT_NE(nullptr, source);

    int32_t 
    ret = event_loop_->SourceAdd(source, hozon::netaos::diag::DOIP_EVENT_READABLE);
    EXPECT_EQ(0, ret);
    ret = event_loop_->SourceRemove(source, 0);
    EXPECT_EQ(0, ret);
    
    ret = event_loop_->SourceAdd(source, hozon::netaos::diag::DOIP_EVENT_READABLE);
    EXPECT_EQ(0, ret);
    ret = event_loop_->SourceRemove(source, 1);// 1,close(fd)
    EXPECT_EQ(0, ret);
    EXPECT_EQ(-1, source->fd);

    ret = event_loop_->SourceRemove(source, 0);
    EXPECT_EQ(-1, ret);

    event_loop_->SourceDestroy(source);
    delete[] data;
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}
TEST(DoipEventLoop, SourceUpdate) {
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    std::shared_ptr<hozon::netaos::diag::DoipEventLoop> event_loop_;
    event_loop_ = std::make_shared<hozon::netaos::diag::DoipEventLoop>();

    uint8_t *data = new uint8_t[4];
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::doip_event_source_t* source = event_loop_->SourceCreate(fd, DoipEventLoop_test_call_back, data);
    ASSERT_NE(nullptr, source);
    
    int32_t 
    ret = event_loop_->SourceAdd(source, hozon::netaos::diag::DOIP_EVENT_READABLE);
    EXPECT_EQ(0, ret);
    ret = event_loop_->SourceUpdate(source, hozon::netaos::diag::DOIP_EVENT_WRITABLE);
    EXPECT_EQ(0, ret);
    ret = event_loop_->SourceRemove(source, 1);// 1,close(fd)
    EXPECT_EQ(0, ret);

    event_loop_->SourceDestroy(source);
    delete[] data;
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}
TEST(DoipEventLoop, Dispatch) {
    hozon::netaos::diag::DoipSelect::Instance()->Create();
    std::shared_ptr<hozon::netaos::diag::DoipEventLoop> event_loop_;
    event_loop_ = std::make_shared<hozon::netaos::diag::DoipEventLoop>();

    uint8_t *data = new uint8_t[4];
    int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
    hozon::netaos::diag::doip_event_source_t* source = event_loop_->SourceCreate(fd, DoipEventLoop_test_call_back, data);
    ASSERT_NE(nullptr, source);

    int32_t 
    ret = event_loop_->SourceAdd(source, hozon::netaos::diag::DOIP_EVENT_READABLE);
    EXPECT_EQ(0, ret);

    ret = event_loop_->Dispatch(1000);
    EXPECT_EQ(0, ret);

    ret = event_loop_->SourceRemove(source, 0);
    EXPECT_EQ(0, ret);
    event_loop_->SourceDestroy(source);
    delete[] data;
    hozon::netaos::diag::DoipSelect::Instance()->Destroy();
}

