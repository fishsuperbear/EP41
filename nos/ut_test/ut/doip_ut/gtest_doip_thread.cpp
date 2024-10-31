

#include "gtest/gtest.h"
#include "base/doip_thread.h"
void doipthread_test_call_back(void *) {
    //printf("doipthread_test_call_back\n");
}
TEST(DoipThread, DoipThreadCreate) {
    EXPECT_TRUE(nullptr == hozon::netaos::diag::DoipThread::Instance()->DoipThreadCreate(nullptr, nullptr, "gtest_thread"));
    hozon::netaos::diag::doip_thread_t* thread = hozon::netaos::diag::DoipThread::Instance()->DoipThreadCreate(doipthread_test_call_back, nullptr, "gtest_thread");
    EXPECT_TRUE(nullptr != thread);
    EXPECT_TRUE(-1 == hozon::netaos::diag::DoipThread::Instance()->DoipThreadWait(nullptr));
}
TEST(DoipThread, DoipThreadRelease) {
    hozon::netaos::diag::doip_thread_t* thread = hozon::netaos::diag::DoipThread::Instance()->DoipThreadCreate(doipthread_test_call_back, nullptr, "gtest_thread");
    EXPECT_TRUE(nullptr != thread);
    if (nullptr != thread) {
        usleep(5*1000);//等待线程拉起
        hozon::netaos::diag::DoipThread::Instance()->DoipThreadRelease(thread);
        //EXPECT_TRUE(nullptr == thread);
    }
}
TEST(DoipThread, DoipThreadGetName) {
    hozon::netaos::diag::doip_thread_t* thread = hozon::netaos::diag::DoipThread::Instance()->DoipThreadCreate(doipthread_test_call_back, nullptr, "gtest_thread");
    EXPECT_TRUE(nullptr != thread);
    EXPECT_TRUE("unknown" == hozon::netaos::diag::DoipThread::Instance()->DoipThreadGetName(nullptr));
    EXPECT_TRUE("gtest_thread" == hozon::netaos::diag::DoipThread::Instance()->DoipThreadGetName(thread));
}
TEST(DoipThread, DoipThreadGetId) {
    hozon::netaos::diag::doip_thread_t* thread = hozon::netaos::diag::DoipThread::Instance()->DoipThreadCreate(doipthread_test_call_back, nullptr, "gtest_thread");
    EXPECT_TRUE(nullptr != thread);
    usleep(5*1000);//等待线程拉起
    EXPECT_TRUE(-1 == hozon::netaos::diag::DoipThread::Instance()->DoipThreadGetId(nullptr));
    EXPECT_EQ(getpid(), hozon::netaos::diag::DoipThread::Instance()->DoipThreadGetId(thread));
}

