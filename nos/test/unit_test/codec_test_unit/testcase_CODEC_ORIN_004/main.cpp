#include <nvmedia_iep.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <signal.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <thread>
#include "gtest/gtest.h"
#include "codec/include/codec_def.h"
#include "codec/include/encoder.h"
#include "codec/include/encoder_factory.h"
#include "codec/include/scibuf_utils.h"

using hozon::netaos::codec::Encoder;
using hozon::netaos::codec::EncoderFactory;

std::string strfile = "./NVIDIA_LOGO_SLIGHT_PUSH_track1.265";
std::string dstfile = "./NVIDIA_LOGO_SLIGHT_PUSH_track1.yuv";
bool iscallback = true;

class OrinCodecEncodeTest {

   public:
    OrinCodecEncodeTest(bool iscallback = false) : iscallback(iscallback){};

    ~OrinCodecEncodeTest(){};

    std::unique_ptr<Encoder> CreateEncoder() {
        auto encoder_uptr = EncoderFactory::Create(hozon::netaos::codec::kDeviceType_NvMedia);
        if (iscallback) {

            encoder_uptr->Init("");
        } else {
            encoder_uptr->Init("");
        }

        return encoder_uptr;
    }

    bool iscallback = false;
};

class EncoderTest : public ::testing::Test {
   protected:
    void SetUp() override { instance = new OrinCodecEncodeTest(); }

    void TearDown() override { delete instance; }

   protected:
    OrinCodecEncodeTest* instance;

   public:
};

TEST(OrinCodecEncodeTest, EncoderCreateTest) {}

TEST_F(EncoderTest, NV12_TO_H265_SCIBUF) {}

TEST_F(EncoderTest, RGB_TO_H265_SCIBUF) {}

// TEST_F(EncoderTest, RGB_TO_H265_SCIBUF) {}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();
    return res;
}
