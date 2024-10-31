#ifndef CODEC_CODEC_ERROR_DOMAIN_H_
#define CODEC_CODEC_ERROR_DOMAIN_H_
#pragma once

#ifdef __cplusplus
namespace hozon {
namespace netaos {
namespace codec {
#endif

enum CodecErrc {
    kEncodeSuccess = 0,
    kDecodeSuccess = 0,
    kDecodeInvalidFrame = 1,

    kDeviceNotSupported = 0xffff0fff,
    kCodecNotSupported,

    // Encode error code start from 0xffff 0000 to 0xffff 0fff
    kEncodeFailed = 0xffff1fff,
    kEncodeInitError,
    kEncodeNotInited,
    kEncodeGetParamFailed,
    kEncodeNotImplemented,

    // Decode error code start from 0xffff 1000 to 0xffff 1fff
    kDecodeFailed = 0xffff2fff,
    kDecodeInitError,
    kDecodeNotInited,
    kDecodeGetParamFailed,
    kDecodeNotImplemented,
};

#define CUDA_CHECK_RET_NULL(call)                                                                     \
    do {                                                                                              \
        cudaError_t status = (call);                                                                  \
        if (cudaSuccess != status) {                                                                  \
            fprintf(stdout, "Cuda error in file '%s' in line %d: %d.\n", __FILE__, __LINE__, status); \
        }                                                                                             \
    } while (0)

#define CUDA_CHECK_RET(call)                                                                          \
    do {                                                                                              \
        cudaError_t status = (call);                                                                  \
        if (cudaSuccess != status) {                                                                  \
            fprintf(stdout, "Cuda error in file '%s' in line %d: %d.\n", __FILE__, __LINE__, status); \
            return false;                                                                             \
        }                                                                                             \
    } while (0)

#ifdef __cplusplus
}
}
}
#endif

#endif