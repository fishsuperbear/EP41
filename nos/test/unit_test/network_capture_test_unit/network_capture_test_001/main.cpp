#include <iostream>
#include <filesystem>
#include "gtest/gtest.h"
#include "network_capture/include/network_capture.h"

using namespace std;
using namespace hozon::netaos::network_capture;


sig_atomic_t g_stopFlag = 0;

class Network_captureFuncTest : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {

    }
    static void TearDownTestSuite() {

    }
    void SetUp() override {
       network_capture = std::make_unique<NetworkCapture>();
    }
    void TearDown() override {}

 protected:
    std::unique_ptr<NetworkCapture> network_capture;
};

TEST_F(Network_captureFuncTest, Init) {
    ASSERT_EQ(network_capture->Init(), true);
}

TEST_F(Network_captureFuncTest, Run) {
    ASSERT_EQ(network_capture->Run(), true);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000 * 10));
}

TEST_F(Network_captureFuncTest, Stop) {
    ASSERT_EQ(network_capture->Stop(), true);
}

TEST_F(Network_captureFuncTest, DeInit) {
    ASSERT_EQ(network_capture->DeInit(), true);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}