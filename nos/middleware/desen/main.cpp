#include <string>
// #include "h265ToYuv.h"
// #include "YuvToH265.h"
// #include "Yuv2Rgb.h"
#include <fstream>
#include "desen_process.h"

using namespace std;

std::string dstfile = "./NVIDIA_LOGO_SLIGHT_PUSH_track1.yuv";
std::string strfile = "./NVIDIA_LOGO_SLIGHT_PUSH_track1.265";

static std::unique_ptr<std::ofstream> f;

int main(int argc, char* argv[]) {

    uint8_t* bits;
    uint32_t y_size = 1920 * 1080 * 8;
    uint32_t uv_size = (1920 / 2) * (1080 / 2) * 8;

    uint32_t framesize = y_size + 2 * uv_size;

    bits = (uint8_t*)malloc(framesize);
    if (!bits) {
        return -1;
    }

    FILE* file = fopen(argv[1], "rb");
    if (!file) {
        return -1;
    }

    size_t len = fread(bits, 1, framesize, file);
    std::string input_buff(reinterpret_cast<char*>(bits), reinterpret_cast<char*>(bits) + len);
    // std::string file_path = "camyyyy.265";
    // f.reset(new std::ofstream(file_path, std::ios::binary | std::ios::out));
    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////// 111  H265ToYuv //////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////

    // H265ToYuv h265toyub;
    // hozon::netaos::codec::DecoderBufNvSpecific output_buff;
    // h265toyub.Process(input_buff, output_buff);

    ////////////////////////// 22222222  YuvToRGB //////////////////////////////////////
    // std::string configFile("./config.yaml");
    // Yuv2Rgb yuvtoRgb(configFile);
    // yuvtoRgb.process();

    ////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////// 44444  YuvToH265 ////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////

    // cout << "yuvdata convert form output_buff start" << endl;
    // int frame_size = 1920 *1080 *3 / 2;
    // std::string yuv_cpu_buf(reinterpret_cast<char*>(output_buff.buf_obj), reinterpret_cast<char*>(output_buff.buf_obj) + frame_size);
    // YuvToH265 yuv2h265(1920, 1080);
    // yuv2h265.init();
    // std::string dest;
    // yuv2h265.Process(yuv_cpu_buf, dest);

    ///// 5555555555  desenProcess /////////////////////////

    DesenProcess pro(1920, 1080);
    std::string output;
    pro.Process(input_buff, output);
    printf("output size=%d\n", output.size());
    // *f << output;
    // pro.process();

    return 0;
}
