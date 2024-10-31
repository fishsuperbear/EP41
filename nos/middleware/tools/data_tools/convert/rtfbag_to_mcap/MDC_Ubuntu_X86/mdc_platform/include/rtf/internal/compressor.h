/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: ddt compressor
*/
#ifndef DDT_COMPRESSOR_H
#define DDT_COMPRESSOR_H

#include <cstdint>

namespace mdc {
namespace ddt {
enum class CompressionType : uint32_t {
    NONE = 0,
    GZIP = 1,
    ZLIB = 2
};

enum class CompressorAction : uint32_t {
    COMPRESS = 0,
    DECOMPRESS
};

enum class CompressorRetCode : uint32_t {
    RET_SUCCESS = 0,
    RET_GZIP_COMPRESSOR_INIT_FAILED = 1,
    RET_ZLIB_COMPRESSOR_INIT_FAILED = 2,
    RET_GZIP_DECOMPRESSOR_INIT_FAILED = 3,
    RET_ZLIB_DECOMPRESSOR_INIT_FAILED = 4,
    RET_PARAM_ERROR = 5,
    RET_NOT_SUPPORT_TYPE = 6,
    RET_COMPRESS_ERROR = 7,
    RET_COMPRESSOR_DEINIT_FAILED = 8,
    RET_DECOMPRESSOR_DEINIT_FAILED = 9,
    RET_MEMCPY_FAILED = 10
};

class Compressor {
public:
    explicit Compressor(const CompressionType& type, const CompressorAction& action);
    virtual ~Compressor();
    int32_t TransferStream(const uint8_t* inStream, const uint64_t &inputLength,
                           uint8_t* &outStream, uint64_t &outputLength);
    int32_t Init();
    void DeInit() noexcept;

private:
    Compressor(const Compressor&) = delete;
    Compressor& operator=(const Compressor&) = delete;
    Compressor(Compressor&&) = delete;
    Compressor& operator=(Compressor&&) = delete;

    int32_t StreamInit();
    int32_t StreamEnd();
    int32_t CompressInit();
    int32_t DecompressInit();

    int32_t Compress(const uint8_t* inStream, const uint64_t &inputLength,
                     uint8_t* &outStream, uint64_t &outputLength);
    int32_t Decompress(const uint8_t* inStream, const uint64_t &inputLength,
                       uint8_t* &outStream, uint64_t &outputLength);

    CompressionType compressionType_;
    CompressorAction actionType_;
    void* stream_;
};
}
}
#endif
