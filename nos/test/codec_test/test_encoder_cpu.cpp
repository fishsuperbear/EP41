/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-10-16 16:43:35
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-11-02 20:15:04
 * @FilePath: /nos/test/codec_test/test_encoder_cpu.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <fstream>
#include <iostream>
#include <list>

#include "codec/include/codec_error_domain.h"
#include "codec/include/encoder.h"
#include "codec/include/encoder_factory.h"
#include "log/include/logging.h"
using namespace hozon::netaos::codec;
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "parameter count error.\n";
        return -1;
    }

    std::string yuv_file = argv[2];
    std::string h265_file = argv[3];
    std::string cfg_file = argv[1];

    std::cout << "encoder start.\n";
    hozon::netaos::log::InitLogging("CODEC_TEST",                         // the id of application
                                    "codec_test",                         // the log id of application
                                    hozon::netaos::log::LogLevel::kInfo,  // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE,   // the output log mode
                                    "./",                                 // the log file directory, active when output log to file
                                    10,                                   // the max number log file , active when output log to file
                                    20                                    // the max size of each  log file , active when output log to file
    );

    auto encoder_uptr = EncoderFactory::Create(hozon::netaos::codec::kDeviceType_Cpu);
    if (cfg_file == "JPEG") {
        EncodeInitParam param;
        param.width = 3840;
        param.height = 2160;
        param.yuv_type = hozon::netaos::codec::YuvType::kYuvType_YUVJ420P;
        CodecErrc res = encoder_uptr->Init(param);
        std::cout << "init res: " << static_cast<int>(res) << std::endl;
        if (res != 0) {
            return 0;
        }
        FILE* file = fopen(reinterpret_cast<const char*>(yuv_file.c_str()), "rb");
        if (!file) {
            std::cout << "Init: Failed to open stream " << yuv_file << std::endl;
            return -1;
        }
        fseek(file, 0, SEEK_END);
        // 获取文件大小
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        uint8_t* bits = (uint8_t*)malloc(size);  // 183318
        if (!bits) {
            std::cout << "Decode_orig: Failed allocating memory for file buffer" << std::endl;
            return -1;
        }
        size_t len = fread(bits, 1, size, file);
        std::cout << "Init: fread len    " << len << std::endl;
        // std::string input_buff(reinterpret_cast<char*>(bits), len);
        auto input_buff = new std::string("");
        input_buff->append((char*)bits, len);
        FrameType frame_type;
        std::string h265_buf;
        res = encoder_uptr->Process(*input_buff, h265_buf, frame_type);
        std::cout << "process "
                  << " frames. result: " << res << ", yuv_bytes: " << input_buff->size() << " h265_bytes: " << h265_buf.size() << " frame_type: " << frame_type << std::endl;
        if (res < 0) {
            std::cout << "failed to encode the frame.\n";
        } else {
            std::ofstream ofs(h265_file, std::ios::binary | std::ios::app | std::ios::out);
            ofs.write(h265_buf.data(), h265_buf.size());
        }
        free(bits);
    } else if (cfg_file == "YUV420P") {
        EncodeInitParam param;
        param.width = 1920;
        param.height = 1080;
        param.yuv_type = hozon::netaos::codec::YuvType::kYuvType_YUV420P;
        CodecErrc res = encoder_uptr->Init(param);
        std::cout << "init res: " << static_cast<int>(res) << std::endl;
        if (res != 0) {
            return 0;
        }
        int frame_num = 0;
        std::ifstream ifs(yuv_file);
        if (!ifs) {
            std::cout << "cannot opep yuv file: " << yuv_file << std::endl;
        }
        const size_t yuv_frame_size = 1920 * 1080 * 3 / 2;
        {
            ifs.seekg(0, std::ios::end);
            size_t size = ifs.tellg();
            if (size % (yuv_frame_size) != 0) {
                std::cout << "yuv file size is not error. expected: 1920 * 1080 * 3 /2.\n";
                return -1;
            }
            frame_num = size / (yuv_frame_size);
            ifs.seekg(0, std::ios::beg);
        }
        std::cout << "frame_num: " << frame_num << " in input file: " << yuv_file << std::endl;
        for (int i = 0; i < frame_num; ++i) {
            std::vector<uint8_t> yuv_buf;
            std::vector<uint8_t> h265_buf;
            yuv_buf.resize(yuv_frame_size);
            if (!ifs.read(reinterpret_cast<char*>(yuv_buf.data()), yuv_buf.size())) {
                std::cout << "failed to read the " << i << " th yuv frame from file.\n";
                break;
            }

            FrameType frame_type;
            res = encoder_uptr->Process(yuv_buf, h265_buf, frame_type);
            std::cout << "process " << static_cast<int>(i) << " frames. result: " << res << ", yuv_bytes: " << yuv_buf.size() << " h265_bytes: " << h265_buf.size() << " frame_type: " << frame_type
                      << std::endl;
            if (res < 0) {
                std::cout << "failed to encode the " << i << " th frame.\n";
                break;
            } else {
                std::ofstream ofs(h265_file, std::ios::binary | std::ios::app | std::ios::out);
                ofs.write(reinterpret_cast<char*>(h265_buf.data()), h265_buf.size());
            }
        }
    }

    return 0;
}
