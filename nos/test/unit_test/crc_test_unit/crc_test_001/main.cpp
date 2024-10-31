#include <iostream>
#include <crc.h>
#include "gtest/gtest.h"

using namespace std;

sig_atomic_t g_stopFlag = 0;

uint8_t data[10][10] = {{0x00, 0x00, 0x00, 0x00}, {0xf2, 0x01, 0x83},      {0x0f, 0xaa, 0x00, 0x55}, {0x00, 0xff, 0x55, 0x11}, {0x33, 0x22, 0x55, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
                        {0x92, 0x6B, 0x55},       {0xff, 0xff, 0xff, 0xff}};

uint32_t len[10] = {4, 3, 4, 4, 9, 3, 4};

uint64_t ans[10][10] = {
    {0x59,0x37,0x79,0xB8,0xCB,0x8C,0x74},
    {0x12,0xC2,0xC6,0x77,0x11,0x33,0x6C},
    {0x84C0,0xD374,0x2023,0xB8F9,0xF53F,0x0745,0x1D0F},
    {0x0000,0xC2E1,0x0BE3,0x6CCF,0xAE98,0xE24E,0x9401},
    {0x2144DF1C,0x24AB9D77,0xB6C9B287,0x32A06212,0xB0AE863D,0x9CDEA29B,0xFFFFFFFF},
    {0x6FB32240,0x4F721A25,0x20662DF8,0x9BD7996E,0xA65A343D,0xEE688A78,0xFFFFFFFF},
    {0xF4A586351E1B9F4B,0x319C27668164F1C6,0x54C5D0F7667C1575,0xA63822BE7E0704E6,0x701ECEB219A8E5D5,0x5FAA96A9B59F3E4E,0xFFFFFFFF00000000}
};
class CrcFuncTest : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {

    }
    static void TearDownTestSuite() {

    }
    void SetUp() override {}
    void TearDown() override {}

 protected:

};

TEST_F(CrcFuncTest, CRC8) {
    for (uint8_t i = 0; i < 7; i++) {
        EXPECT_EQ(ans[0][i], Crc_CalculateCRC8(data[i], len[i], 0, true));
    }
}

TEST_F(CrcFuncTest, CRC8H2F) {
    for (uint8_t i = 0; i < 7; i++) {
        EXPECT_EQ(ans[1][i], Crc_CalculateCRC8H2F(data[i], len[i], 0, true));
    }
}

TEST_F(CrcFuncTest, CRC16) {
    for (uint8_t i = 0; i < 7; i++) {
        EXPECT_EQ(ans[2][i], Crc_CalculateCRC16(data[i], len[i], 0, true));
    }
}

TEST_F(CrcFuncTest, CRC16ARC) {
    for (uint8_t i = 0; i < 7; i++) {
        EXPECT_EQ(ans[3][i], Crc_CalculateCRC16ARC(data[i], len[i], 0, true));
    }
}

TEST_F(CrcFuncTest, CRC32) {
    for (uint8_t i = 0; i < 7; i++) {
        EXPECT_EQ(ans[4][i], Crc_CalculateCRC32(data[i], len[i], 0, true));
    }
}

TEST_F(CrcFuncTest, CRC32P4) {
    for (uint8_t i = 0; i < 7; i++) {
        EXPECT_EQ(ans[5][i], Crc_CalculateCRC32P4(data[i], len[i], 0, true));
    }
}

TEST_F(CrcFuncTest, CRC64) {
    for (uint8_t i = 0; i < 7; i++) {
        EXPECT_EQ(ans[6][i], Crc_CalculateCRC64(data[i], len[i], 0, true));
    }
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}