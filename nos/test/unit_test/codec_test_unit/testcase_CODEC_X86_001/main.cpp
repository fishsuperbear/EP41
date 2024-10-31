#include <iostream>
#include <memory>
#include "gtest/gtest.h"

class OrinCodecEncodeTest {

   public:
    OrinCodecEncodeTest(){};
    ~OrinCodecEncodeTest(){};

    int myadd(int a, int b) { return a + b; }
};

class MMMTest : public ::testing::Test {
   protected:
    void SetUp() override { instance = new OrinCodecEncodeTest(); }

    void TearDown() override { delete instance; }

   protected:
    OrinCodecEncodeTest* instance;

   public:
};

TEST(OrinCodecEncodeTest, H265_TO_NV12_SCIBUF) {
    OrinCodecEncodeTest* instance = new OrinCodecEncodeTest();
    EXPECT_NE(instance, nullptr);
    int res = instance->myadd(2, 3);
    EXPECT_EQ(res, 5);
}

TEST(OrinCodecEncodeTest, H265_TO_YUYV_SCIBUF) {
    OrinCodecEncodeTest* instance = new OrinCodecEncodeTest();
    EXPECT_NE(instance, nullptr);
    int res = instance->myadd(2, 3);
    EXPECT_EQ(res, 5);
}

int main(int argc, char* argv[]) {
    int aaa = 10;
    std::cout << aaa << std::endl;
    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();
    return res;
}
