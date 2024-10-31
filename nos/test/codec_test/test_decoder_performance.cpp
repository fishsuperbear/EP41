#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <thread>

#include "nvmedia_core.h"
// #include "nvscierror.h"
#include "nvscibuf.h"

#include "codec/include/decoder.h"
#include "codec/include/decoder_factory.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "proto/soc/sensor_image.pb.h"
#include "sys/time.h"
#include "tools/data_tools/bag/include/reader.h"
#include "log/include/logging.h"

using namespace hozon::netaos::bag;

using hozon::netaos::codec::Decoder;
using hozon::netaos::codec::DecoderFactory;

#define MAX_NUM_SURFACES (3U)
#define MAX_PLANE_COUNT 3

typedef struct {
    NvSciBufType bufType;
    uint64_t size;
    uint32_t planeCount;
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planeWidths[MAX_NUM_SURFACES];
    uint32_t planeHeights[MAX_NUM_SURFACES];
    uint32_t planePitches[MAX_NUM_SURFACES];
    uint32_t planeBitsPerPixels[MAX_NUM_SURFACES];
    uint32_t planeAlignedHeights[MAX_NUM_SURFACES];
    uint64_t planeAlignedSizes[MAX_NUM_SURFACES];
    uint8_t planeChannelCounts[MAX_NUM_SURFACES];
    uint64_t planeOffsets[MAX_NUM_SURFACES];
    uint64_t topPadding[MAX_NUM_SURFACES];
    uint64_t bottomPadding[MAX_NUM_SURFACES];
    bool needSwCacheCoherency;
    NvSciBufAttrValColorFmt planeColorFormats[MAX_NUM_SURFACES];
} BufferAttrs;

/* Enum specifying the different ways to read/write surface data from/to file */
typedef enum {
    /* Use NvSci buffer r/w functionality */
    FILE_IO_MODE_NVSCI = 0,
    /* Copy surface data line-by-line discarding any padding */
    FILE_IO_MODE_LINE_BY_LINE,
} FileIOMode;

typedef struct {
    void* buffer;
    uint64_t size;
    uint32_t planeCount;
    void* planePtrs[MAX_PLANE_COUNT];
    uint32_t planeSizes[MAX_PLANE_COUNT];
    uint32_t planePitches[MAX_PLANE_COUNT];
} PixelDataBuffer;

bool IsInterleavedYUV(NvSciBufAttrValColorFmt colorFmt) {
    switch (colorFmt) {
        case NvSciColor_Y8U8Y8V8:
        case NvSciColor_Y8V8Y8U8:
        case NvSciColor_U8Y8V8Y8:
        case NvSciColor_V8Y8U8Y8:
            return true;
        default:
            return false;
    }
}

int AllocatePixelDataBuffer(PixelDataBuffer* px, NvSciBufObj sciBuf, FileIOMode mode) {
    NvSciError sciResult = NvSciError_Success;

    NvSciBufAttrList attrList;
    sciResult = NvSciBufObjGetAttrList(sciBuf, &attrList);
    if (sciResult != NvSciError_Success) {
        std::cout << "Failed to get buffer attribute list" << std::endl;
        return 1;
    }

    enum { PLANE_COUNT_ATTR_IDX, PLANE_COLORFMT_ATTR_IDX, PLANE_WIDTH_ATTR_IDX, PLANE_HEIGHT_ATTR_IDX, PLANE_BPP_ATTR_IDX, ATTR_COUNT };

    NvSciBufAttrKeyValuePair attrs[ATTR_COUNT];
    attrs[PLANE_COUNT_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneCount;
    attrs[PLANE_COLORFMT_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneColorFormat;
    attrs[PLANE_WIDTH_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneWidth;
    attrs[PLANE_HEIGHT_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneHeight;
    attrs[PLANE_BPP_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneBitsPerPixel;
    sciResult = NvSciBufAttrListGetAttrs(attrList, attrs, ATTR_COUNT);
    if (sciResult != NvSciError_Success) {
        std::cout << "Failed to get buffer attributes" << std::endl;
        return 1;
    }

    px->size = 0;
    px->planeCount = *((const uint32_t*)attrs[PLANE_COUNT_ATTR_IDX].value);

    for (uint32_t i = 0; i < px->planeCount; ++i) {
        uint32_t width = ((const uint32_t*)attrs[PLANE_WIDTH_ATTR_IDX].value)[i];
        uint32_t height = ((const uint32_t*)attrs[PLANE_HEIGHT_ATTR_IDX].value)[i];
        uint32_t bpp = ((const uint32_t*)attrs[PLANE_BPP_ATTR_IDX].value)[i];
        uint64_t pitchBits = width * bpp;
        px->planeSizes[i] = pitchBits * height / 8;
        px->planePitches[i] = pitchBits / 8;
        px->size += px->planeSizes[i];
    }

    /* NvSciBufObjGet/PutPixels() requires three separate planes for interleaved and semiplanar YUV
     * surfaces. Do special handling for them.*/
    if (mode == FILE_IO_MODE_NVSCI) {
        if (px->planeCount == 1 && IsInterleavedYUV(((const NvSciBufAttrValColorFmt*)attrs[PLANE_COLORFMT_ATTR_IDX].value)[0])) {
            px->planeCount = 3;
            uint32_t fullSize = px->planeSizes[0];
            px->planeSizes[0] = fullSize / 2;
            px->planeSizes[1] = fullSize / 4;
            px->planeSizes[2] = fullSize / 4;
            uint32_t fullPitch = px->planePitches[0];
            px->planePitches[0] = fullPitch / 2;
            px->planePitches[1] = fullPitch / 4;
            px->planePitches[2] = fullPitch / 4;
        } else if (px->planeCount == 2) {
            px->planeCount = 3;
            uint32_t fullChromaSize = px->planeSizes[1];
            px->planeSizes[1] = fullChromaSize / 2;
            px->planeSizes[2] = fullChromaSize / 2;
            uint32_t fullChromaPitch = px->planePitches[1];
            px->planePitches[1] = fullChromaPitch / 2;
            px->planePitches[2] = fullChromaPitch / 2;
        }
    }

    px->buffer = malloc(px->size);

    px->planePtrs[0] = px->buffer;
    for (uint32_t i = 1; i < px->planeCount; ++i) {
        px->planePtrs[i] = ((char*)px->planePtrs[i - 1]) + px->planeSizes[i - 1];
    }

    return 0;
}

int ReadWritePixelDataLineByLine(NvSciBufObj sciBuf, PixelDataBuffer* px, bool write) {
    NvSciError sciResult = NvSciError_Success;

    void* sciBufPtr;
    sciResult = NvSciBufObjGetCpuPtr(sciBuf, &sciBufPtr);
    if (sciResult != NvSciError_Success) {
        std::cout << "Failed to get buffer CPU pointer" << std::endl;
        return 1;
    }

    NvSciBufAttrList attrList;
    sciResult = NvSciBufObjGetAttrList(sciBuf, &attrList);
    if (sciResult != NvSciError_Success) {
        std::cout << "Failed to get buffer attribute list" << std::endl;
        return 1;
    }

    enum { PLANE_OFFSET_ATTR_IDX, PLANE_PITCH_ATTR_IDX, ATTR_COUNT };

    NvSciBufAttrKeyValuePair attrs[ATTR_COUNT];
    attrs[PLANE_OFFSET_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneOffset;
    attrs[PLANE_PITCH_ATTR_IDX].key = NvSciBufImageAttrKey_PlanePitch;
    sciResult = NvSciBufAttrListGetAttrs(attrList, attrs, ATTR_COUNT);
    if (sciResult != NvSciError_Success) {
        std::cout << "Failed to get buffer attributes" << std::endl;
        return 1;
    }

    for (uint32_t plane = 0; plane < px->planeCount; ++plane) {
        uint64_t sciBufPlaneOffset = ((const uint64_t*)attrs[PLANE_OFFSET_ATTR_IDX].value)[plane];
        uint32_t sciBufPlanePitch = ((const uint32_t*)attrs[PLANE_PITCH_ATTR_IDX].value)[plane];
        uint32_t planeHeight = px->planeSizes[plane] / px->planePitches[plane];
        for (uint32_t line = 0; line < planeHeight; ++line) {
            char* sciBufLine = (char*)sciBufPtr + sciBufPlaneOffset + line * sciBufPlanePitch;
            char* pxBufLine = (char*)px->planePtrs[plane] + line * px->planePitches[plane];
            if (write) {
                memcpy(sciBufLine, pxBufLine, px->planePitches[plane]);
            } else {
                memcpy(pxBufLine, sciBufLine, px->planePitches[plane]);
            }
        }
    }

    return 0;
}

void DeallocatePixelDataBuffer(PixelDataBuffer* px) {
    if (px->buffer) {
        free(px->buffer);
        px->buffer = NULL;
    }
}

int WriteBufferToFile(NvSciBufObj buffer, const std::string& filename, FileIOMode mode) {
    int retval = 0;
    NvSciError sciResult = NvSciError_Success;
    FILE* file = NULL;
    size_t bytesWritten = 0;
    PixelDataBuffer px;
    memset(&px, 0, sizeof(px));
    if (AllocatePixelDataBuffer(&px, buffer, mode) != 0) {
        std::cout << "Failed to allocate pixel data buffer" << std::endl;
        retval = 1;
        goto WriteBufferToFileEnd;
    }

    file = fopen(filename.data(), "wb");
    if (!file) {
        printf("Failed to open file %s", filename.data());
        retval = 1;
        goto WriteBufferToFileEnd;
    }

    switch (mode) {
        case FILE_IO_MODE_NVSCI:
            sciResult = NvSciBufObjGetPixels(buffer, NULL, px.planePtrs, px.planeSizes, px.planePitches);
            if (sciResult != NvSciError_Success) {
                printf("Failed to read data from buffer using NvSci");
                retval = 1;
                goto WriteBufferToFileEnd;
            }
            break;

        case FILE_IO_MODE_LINE_BY_LINE:
            if (ReadWritePixelDataLineByLine(buffer, &px, false) != 0) {
                printf("Failed to read data from buffer line-by-line");
                retval = 1;
                goto WriteBufferToFileEnd;
            }
            break;
    }

    bytesWritten = fwrite(px.buffer, 1, px.size, file);
    if (bytesWritten != px.size) {
        printf("Expected to write %u bytes, did %u", px.size, bytesWritten);
        retval = 1;
        goto WriteBufferToFileEnd;
    }

WriteBufferToFileEnd:
    DeallocatePixelDataBuffer(&px);

    if (file) {
        fclose(file);
    }

    return retval;
}

void create_decoder(std::list<hozon::soc::CompressedImage>* data_list,
                    std::string topic_name, std::string path) {
  std::unordered_map<std::string, std::string> config;
  auto decoder_uptr = DecoderFactory::Create(hozon::netaos::codec::kDeviceType_NvMedia);
  decoder_uptr->Init("");

  // std::ofstream outh265("my_reader.h265",
  //                       std::ios::binary | std::ios::app | std::ios::out);
  // std::ofstream outyuv("my_reader.yuv",
  //                      std::ios::binary | std::ios::app | std::ios::out);

  std::cout << "begin read " << topic_name << std::endl;
  std::chrono::time_point<std::chrono::steady_clock> fps_control =
      std::chrono::steady_clock::now();
  while (true) {
    if (data_list->empty()) {
      usleep(1000);
      continue;
    }

    auto proto_data = data_list->front();
    data_list->pop_front();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - fps_control)
               .count() < 40) {
      usleep(1000);
    }
    fps_control = std::chrono::steady_clock::now();

    // int data_size = proto_data.data().size();
    // const char* data_ = proto_data.data().c_str();
    // outh265.write(data_, data_size);
    // std::string out_buff;
    timeval after, before;
    gettimeofday(&before, NULL);

    // 解码
    hozon::netaos::codec::DecoderBufNvSpecific out_buff;
    decoder_uptr->Process(proto_data.data(), out_buff);

    static int write_count = 0;
    write_count++;
    std::string filename = std::string("test_") + std::to_string(write_count) + ".yuv";
    WriteBufferToFile((NvSciBufObj)(out_buff.buf_obj), filename, FILE_IO_MODE_NVSCI);

    gettimeofday(&after, NULL);
    // int time_diff = (after.tv_sec - before.tv_sec) * 1000 + (after.tv_usec - before.tv_usec) / 1000;
    // std::cout << "input data_size= " << data_size
    //           << " output data_size= " << out_buff.size() << " cost "
    //           << time_diff << "ms" << std::endl;
    // outyuv.write(out_buff.c_str(), out_buff.size());
  }

  // outyuv.close();
  // outh265.close();
  std::cout << "end read " << topic_name << std::endl;
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
  if (argc == 2) {
    std::string path = argv[1];
    Reader reader;
    reader.Open(path, "mcap");
    std::vector<std::string> topics;

    // topics.push_back("Camera71");
    // topics.push_back("Camera73");
    // topics.push_back("Camera75");
    // topics.push_back("Camera76");
    // topics.push_back("Camera77");
    // topics.push_back("Camera78");
    // topics.push_back("Camera83");
    topics.push_back("/soc/encoded_camera_1");

    reader.SetFilter(topics);

    // std::list<hozon::soc::CompressedImage> list_camera71, list_camera73,
    //     list_camera75, list_camera76, list_camera77, list_camera78,
    //     list_camera83;

    // std::thread thread_camera71(create_decoder, &list_camera71, "Camera71",
    //                             path);
    // std::thread thread_camera73(create_decoder, &list_camera73, "Camera73",
    //                             path);
    // std::thread thread_camera75(create_decoder, &list_camera75, "Camera75",
    //                             path);
    // std::thread thread_camera76(create_decoder, &list_camera76, "Camera76",
    //                             path);
    // std::thread thread_camera77(create_decoder, &list_camera77, "Camera77",
    //                             path);
    // std::thread thread_camera78(create_decoder, &list_camera78, "Camera78",
    //                             path);
    // std::thread thread_camera83(create_decoder, &list_camera83, "Camera83",
    //                             path);

    std::list<hozon::soc::CompressedImage> list_camera_1;
    std::thread thread_camera_1(create_decoder, &list_camera_1, "/soc/encoded_camera_1", path);

    std::map<std::string, bool> first_iframe_flags;
    first_iframe_flags["/soc/encoded_camera_1"] = false;

    while (reader.HasNext()) {
      std::string frame_id;
      TopicMessage message_vec = reader.ReadNext();
      std::cout << "topic: " << message_vec.topic << std::endl;
      frame_id = message_vec.topic;
      // std::cout << "topic name = " << frame_id << std::endl;
      hozon::soc::CompressedImage proto_data = reader.DeserializeToProto<hozon::soc::CompressedImage>(message_vec);
      if (proto_data.frame_type() == hozon::netaos::codec::kFrameType_I) {
        first_iframe_flags[frame_id] = true;
      }

      if (!first_iframe_flags[frame_id] && (proto_data.frame_type() != hozon::netaos::codec::kFrameType_I)) {
        continue;
      }

      // if (frame_id == "Camera71") {
      //   list_camera71.push_back(proto_data);
      // } else if (frame_id == "Camera73") {
      //   list_camera73.push_back(proto_data);
      // } else if (frame_id == "Camera75") {
      //   list_camera75.push_back(proto_data);
      // } else if (frame_id == "Camera76") {
      //   list_camera76.push_back(proto_data);
      // } else if (frame_id == "Camera77") {
      //   list_camera77.push_back(proto_data);
      // } else if (frame_id == "Camera78") {
      //   list_camera78.push_back(proto_data);
      // } else if (frame_id == "Camera83") {
      //   list_camera83.push_back(proto_data);
      // }

      if (frame_id == "/soc/encoded_camera_1") {
        list_camera_1.push_back(proto_data);
      }
    }

    // thread_camera71.join();
    // thread_camera73.join();
    // thread_camera75.join();
    // thread_camera76.join();
    // thread_camera77.join();
    // thread_camera78.join();
    // thread_camera83.join();
    thread_camera_1.join();

    std::cout << "all camera decode finish." << std::endl;
  }
  return 0;
}
