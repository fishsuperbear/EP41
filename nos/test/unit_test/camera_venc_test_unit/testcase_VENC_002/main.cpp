#include <iostream>
#include <memory>
#include "gtest/gtest.h"

class OrinVencEncodeTest {

   public:
    OrinVencEncodeTest(){};
    ~OrinVencEncodeTest(){};

    int myadd(int a, int b) { return a + b; }
};

class MMMTest : public ::testing::Test {
   protected:
    void SetUp() override { instance = new OrinVencEncodeTest(); }

    void TearDown() override { delete instance; }

   protected:
    OrinVencEncodeTest* instance;

   public:
};

TEST(OrinVencEncodeTest, 2V_NV12_TO_H265_SCIBUF) {
    OrinVencEncodeTest* instance = new OrinVencEncodeTest();
    EXPECT_NE(instance, nullptr);
    int res = instance->myadd(2, 3);
    EXPECT_EQ(res, 5);
}

TEST(OrinVencEncodeTest, 9V_YUYV_TO_H265_SCIBUF) {
    OrinVencEncodeTest* instance = new OrinVencEncodeTest();
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
