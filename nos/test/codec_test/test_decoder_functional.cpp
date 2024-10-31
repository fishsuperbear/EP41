#include <nvscibuf.h>
// #include <nvscibufobj.h>

#include <fstream>
#include <iostream>
#include <list>

#include "codec/include/codec_def.h"
#include "codec/include/decoder.h"
#include "codec/include/decoder_factory.h"
#include "codec/src/nvmedia/utils/scibuf_utils.h"
#include "log/include/logging.h"
// #include "nvmedia_video.h"
using hozon::netaos::codec::Decoder;
using hozon::netaos::codec::DecoderFactory;

std::string strfile = "./NVIDIA_LOGO_SLIGHT_PUSH_track1.265";
std::string dstfile = "./NVIDIA_LOGO_SLIGHT_PUSH_track1.yuv";
bool iscallback = true;
void saveToYuvFile(const char* filename, const hozon::netaos::codec::DecoderBufNvSpecific& output_buff) {
    // size_t bufferSize = NvSciBufObjGetSerializationSize(bufObj);
    // void* buffer = malloc(bufferSize);

    // NvErr status = NvSciBufObjSerialize(bufObj, buffer, bufferSize);
    // if (status != NvSuccess) {
    //     // 序列化失败，处理错误
    // }

    // // 获取YUV数据的指针和大小
    // uint8_t* yuvData = bufObj->base;
    // uint32_t ySize = bufObj->ranges[0].size;
    // uint32_t uSize = bufObj->ranges[1].size / 4;
    // uint32_t vSize = bufObj->ranges[2].size / 4;
    // uint32_t yuvSize = ySize + uSize + vSize;
    // // 创建文件并打开以供写入
    // FILE* file = fopen(filename, "ab+");
    // if (file == NULL) {
    //     printf("Failed to open file for writing.\n");
    //     return;
    // }
    // // 将YUV数据写入文件
    // fwrite(yuvData, sizeof(uint8_t), yuvSize, file);
    // // 关闭文件
    // fclose(file);

    NvMediaRect srcRect;
    srcRect.x0 = 0;
    srcRect.y0 = 0;
    srcRect.x1 = output_buff.displayWidth;
    srcRect.y1 = output_buff.displayHeight;
    NvMediaStatus status = WriteOutput(filename, (NvSciBufObj)(output_buff.buf_obj), true, (output_buff.frame_count == 0) ? false : true, &srcRect);
    if (status != NVMEDIA_STATUS_OK) {
        std::cout << "saveToYuvFile: Write frame to file failed: " << status << std::endl;
    }
    std::cout << "saveToYuvFile: Saving YUV frame  to file" << std::endl;
}
int32_t pfnCbDecoderNvMediaOutput(const hozon::netaos::codec::DecoderBufNvSpecific& output_buff) {
    std::cout << "pfnCbDecoderNvMediaOutput recv data" << std::endl;
    saveToYuvFile(dstfile.c_str(), output_buff);
    return 0;
}

int main(int argc, char** argv) {
    hozon::netaos::log::InitLogging("CODEC_TEST",                                                          // the id of application
                                    "codec_test",                                                          // the log id of application
                                    hozon::netaos::log::LogLevel::kInfo,                                   // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
                                    "./",                                                                  // the log file directory, active when output log to file
                                    10,                                                                    // the max number log file , active when output log to file
                                    20                                                                     // the max size of each  log file , active when output log to file
    );

    auto decoder_uptr = DecoderFactory::Create(hozon::netaos::codec::kDeviceType_NvMedia);
    if (iscallback) {
        hozon::netaos::codec::DecoderNvMediaCb cb;
        cb.CbDecoderNvMediaOutput = pfnCbDecoderNvMediaOutput;
        decoder_uptr->Init(cb);
    } else {
        decoder_uptr->Init("");
    }


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
        if (iscallback) {
            decoder_uptr->Process(input_buff);
        } else {
            hozon::netaos::codec::DecoderBufNvSpecific output_buff;
            decoder_uptr->Process(input_buff, output_buff);
            NvMediaRect srcRect;
            srcRect.x0 = 0;
            srcRect.y0 = 0;
            srcRect.x1 = output_buff.displayWidth;
            srcRect.y1 = output_buff.displayHeight;
            NvSciBufObj bufObj = (NvSciBufObj)(output_buff.buf_obj);
            NvMediaStatus status = WriteOutput(dstfile.c_str(), bufObj, true, (output_buff.frame_count == 0) ? false : true, &srcRect);
            if (status != NVMEDIA_STATUS_OK) {
                std::cout << "saveToYuvFile: Write frame to file failed: " << status << std::endl;
            }
            std::cout << "saveToYuvFile: Saving YUV frame  to file" << std::endl;
        }
    }
    free(bits);
    std::cout << "end test decoder orin." << std::endl;
    return 0;
}
