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
#include "codec/include/codec_error_domain.h"
#include "codec/include/decoder.h"
#include "codec/include/decoder_factory.h"
#include "codec/include/scibuf_utils.h"

// #include "nvmedia_video.h"
using hozon::netaos::codec::Decoder;
using hozon::netaos::codec::DecoderBufNvSpecific;
using hozon::netaos::codec::DecoderFactory;
using hozon::netaos::codec::pfnCbDecoderNvMediaOutput;

std::string strfile1 = "../../res/test_1920x1080.h265";
std::string strfile2 = "../../res/test_3840x2160.h265";
bool iscallback = true;

class OrinCodecDecodeTest {

   public:
    OrinCodecDecodeTest(bool iscallback = false) : iscallback(iscallback){};

    ~OrinCodecDecodeTest(){};

    std::unique_ptr<Decoder> CreateDecoder() {
        auto decoder_uptr = DecoderFactory::Create(hozon::netaos::codec::kDeviceType_NvMedia);
        if (iscallback) {
            decoder_uptr->Init(cb);
        } else {
            decoder_uptr->Init("");
        }
        return decoder_uptr;
    }

    void SetCb(pfnCbDecoderNvMediaOutput cb) { this->cb.CbDecoderNvMediaOutput = cb; }

    void SetImgSize(uint16_t w, uint16_t h) {
        this->w = w;
        this->h = h;
    }

    bool iscallback = false;
    hozon::netaos::codec::DecoderNvMediaCb cb;
    uint16_t w, h;
};

class OrinDecoderTest : public ::testing::Test {
   protected:
    void SetUp() override { instance = new OrinCodecDecodeTest(); }

    void TearDown() override { delete instance; }

   protected:
    OrinCodecDecodeTest* instance;
};

class OrinDecoderTestAsync : public ::testing::Test {
   protected:
    void SetUp() override { instance = new OrinCodecDecodeTest(true); }

    void TearDown() override { delete instance; }

   protected:
    OrinCodecDecodeTest* instance;
};

TEST(OrinCodecDecodeTest, DecoderCreateTest) {
    OrinCodecDecodeTest* instance = new OrinCodecDecodeTest();
    EXPECT_NE(instance, nullptr);
    auto ret = instance->CreateDecoder();
    EXPECT_NE(ret, nullptr);
    ret.reset();
    ret = nullptr;
}

TEST(OrinCodecDecodeTest, DecoderCreateWithCbTest) {
    OrinCodecDecodeTest* instance = new OrinCodecDecodeTest(true);
    EXPECT_NE(instance, nullptr);
    instance->SetCb([](const DecoderBufNvSpecific buf) -> int32_t {});
    auto ret = instance->CreateDecoder();
    EXPECT_NE(ret, nullptr);
    ret.reset();
    ret = nullptr;
}

TEST_F(OrinDecoderTest, H265_TO_NV12_SCIBUF_SYNC) {
    instance->SetImgSize(1920, 1080);
    auto decoder_uptr = instance->CreateDecoder();

    uint8_t* bits;
    uint32_t y_size = instance->w * instance->h * 8;
    uint32_t uv_size = (instance->w / 2) * (instance->h / 2) * 8;

    uint32_t framesize = y_size + 2 * uv_size;
    bits = (uint8_t*)malloc(framesize);
    if (!bits) {
        std::cout << "Decode_orig: Failed allocating memory for file buffer" << std::endl;
    }
    FILE* file = fopen(reinterpret_cast<const char*>(strfile1.c_str()), "rb");
    if (!file) {
        std::cout << "Init: Failed to open stream " << strfile1 << std::endl;
    }
    while (!feof(file)) {
        size_t len;
        len = fread(bits, 1, framesize, file);
        std::cout << "Init: fread len " << len << std::endl;
        std::string input_buff(reinterpret_cast<char*>(bits), reinterpret_cast<char*>(bits) + len);

        hozon::netaos::codec::DecoderBufNvSpecific output_buff;
        auto ret = decoder_uptr->Process(input_buff, output_buff);

        EXPECT_EQ(ret, 0);
        EXPECT_GT(output_buff.img_size, 0);
        EXPECT_NE(output_buff.buf_obj, nullptr);
        EXPECT_NE(output_buff.cuda_ptr, nullptr);
    }
    free(bits);
}

TEST_F(OrinDecoderTestAsync, H265_TO_NV12_SCIBUF_ASYNC) {
    instance->SetImgSize(1920, 1080);
    auto decoder_uptr = instance->CreateDecoder();
    instance->SetCb([](const DecoderBufNvSpecific outbuf) -> int32_t {
        EXPECT_GT(outbuf.img_size, 0);
        // EXPECT_NE(outbuf.buf_obj, nullptr);
        // EXPECT_NE(outbuf.cuda_ptr, nullptr);
        return 0;
    });

    uint8_t* bits;
    uint32_t y_size = instance->w * instance->h * 8;
    uint32_t uv_size = (instance->w / 2) * (instance->h / 2) * 8;

    uint32_t framesize = y_size + 2 * uv_size;
    bits = (uint8_t*)malloc(framesize);
    if (!bits) {
        std::cout << "Decode_orig: Failed allocating memory for file buffer" << std::endl;
    }
    FILE* file = fopen(reinterpret_cast<const char*>(strfile1.c_str()), "rb");
    if (!file) {
        std::cout << "Init: Failed to open stream " << strfile1 << std::endl;
    }
    while (!feof(file)) {
        size_t len;
        len = fread(bits, 1, framesize, file);
        std::cout << "Init: fread len " << len << std::endl;
        std::string input_buff(reinterpret_cast<char*>(bits), reinterpret_cast<char*>(bits) + len);

        auto ret = decoder_uptr->Process(input_buff);

        EXPECT_EQ(ret, 0);
    }
    free(bits);
}

TEST(OrinCodecDecodeTest, H265_TO_YUYV_SCIBUF) {
    OrinCodecDecodeTest* instance = new OrinCodecDecodeTest();
    EXPECT_NE(instance, nullptr);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();
    return res;
}
