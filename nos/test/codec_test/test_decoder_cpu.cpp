

#include <fstream>
#include <iostream>
#include <list>

#include "codec/include/codec_def.h"
#include "codec/include/decoder.h"
#include "codec/include/decoder_factory.h"
#include "log/include/logging.h"
// #include "nvmedia_video.h"
using hozon::netaos::codec::Decoder;
using hozon::netaos::codec::DecoderFactory;

int main(int argc, char** argv) {
    hozon::netaos::log::InitLogging("CODEC_TEST",                         // the id of application
                                    "codec_test",                         // the log id of application
                                    hozon::netaos::log::LogLevel::kInfo,  // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE,   // the output log mode
                                    "./",                                 // the log file directory, active when output log to file
                                    10,                                   // the max number log file , active when output log to file
                                    20                                    // the max size of each  log file , active when output log to file
    );
    std::string strfile = argv[1];
    std::string dstfile = argv[2];
    auto decoder_uptr = DecoderFactory::Create(hozon::netaos::codec::kDeviceType_Cpu);
    decoder_uptr->Init("JPEG");
    std::cout << "begin test decoder orin. " << std::endl;
    uint8_t* bits;
    uint32_t y_size = 1920 * 1080 * 8;
    uint32_t uv_size = (1920 / 2) * (1080 / 2) * 8;

    uint32_t framesize = y_size + 2 * uv_size;
    bits = (uint8_t*)malloc(framesize);
    if (!bits) {
        std::cout << "Decode_orig: Failed allocating memory for file buffer" << std::endl;
        return -1;
    }
    FILE* file = fopen(reinterpret_cast<const char*>(strfile.c_str()), "rb");
    if (!file) {
        std::cout << "Init: Failed to open stream " << strfile << std::endl;
        return -1;
    }
    while (!feof(file)) {
        size_t len;
        len = fread(bits, 1, framesize, file);
        std::cout << "Init: fread len " << len << std::endl;
        std::string input_buff(reinterpret_cast<char*>(bits), reinterpret_cast<char*>(bits) + len);
        std::string output_buff;
        int res = decoder_uptr->Process(input_buff, output_buff);
        if (res < 0) {
            std::cout << "failed to encode the frame.\n";
            break;
        } else {
            std::ofstream ofs(dstfile, std::ios::binary | std::ios::app | std::ios::out);
            ofs.write(output_buff.data(), output_buff.size());
        }
        std::cout << "saveToYuvFile: Saving YUV frame  to file  " << output_buff.size() << std::endl;
    }
    free(bits);
    std::cout << "end test decoder orin." << std::endl;
    return 0;
}
