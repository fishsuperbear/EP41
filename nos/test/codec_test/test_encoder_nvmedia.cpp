#include <signal.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

#include <nvscibuf.h>
#include <nvscisync.h>
#include <nvmedia_iep.h>

#include "codec/include/codec_def.h"
#include "codec/include/codec_error_domain.h"
#include "codec/include/encoder.h"
#include "codec/include/encoder_factory.h"
#include "log/include/logging.h"

using NvSciBufAttrListSptr = std::shared_ptr<NvSciBufAttrList>;
using NvSciSyncAttrListSptr = std::shared_ptr<NvSciSyncAttrList>;
using NvSciBufObjSptr = std::shared_ptr<NvSciBufObj>;
using NvSciBufModuleSptr = std::shared_ptr<NvSciBufModule>;
using NvSciSyncModuleSptr = std::shared_ptr<NvSciSyncModule>;
using NvSciSyncObjSptr = std::shared_ptr<NvSciSyncObj>;
using NvSciSyncCpuWaitContextSptr = std::shared_ptr<NvSciSyncCpuWaitContext>;

#define CHECK_SCIERR_RET(err, op, retval)\
    if (err != NvSciError_Success) {\
        logger_->LogError() << op << " failed. err: " << err;\
        return retval;\
    }

#define CHECK_CODECERR_RET(err, op, retval)\
    if (err != hozon::netaos::codec::kEncodeSuccess) {\
        logger_->LogError() << op << " failed. err: " << static_cast<int64_t>(err);\
        return retval;\
    }

#define CHECK_FALSE_RET(pred, op, retval)\
    if (!pred) {\
        logger_->LogError() << op << " failed.";\
        return retval;\
    }

const int MAX_NUM_SURFACES = 3;
static std::shared_ptr<hozon::netaos::log::Logger> logger_{nullptr};

void InitializeLogging() {

    hozon::netaos::log::InitLogging("CODEC_TEST",                                                          // the id of application
                                    "CODEC_TEST",                                                          // the log id of application
                                    hozon::netaos::log::LogLevel::kDebug,                                   // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE,  // the output log mode
                                    "./",                                                                  // the log file directory, active when output log to file
                                    10,                                                                    // the max number log file , active when output log to file
                                    20                                                                     // the max size of each  log file , active when output log to file
    );

    logger_ = hozon::netaos::log::CreateLogger("CODEC_TEST", "CODEC_TEST", hozon::netaos::log::LogLevel::kDebug);
}

struct NvSciBufAttrListDeleter {
    void operator()(NvSciBufAttrList* p) {
        if (p && *p) {
            NvSciBufAttrListFree(*p);
        }
        if (p) {
            delete p;
        }
    }
};

struct NvSciSyncAttrListDeleter {
    void operator()(NvSciSyncAttrList* p) {
        if (p && *p) {
            NvSciSyncAttrListFree(*p);
        }
        if (p) {
            delete p;
        }
    }
};

struct NvSciBufModuleDeleter {
    void operator()(NvSciBufModule* p) {
        if (p && *p) {
            NvSciBufModuleClose(*p);
        }
        if (p) {
            delete p;
        }
    }
};

struct NvSciSyncModuleDeleter {
    void operator()(NvSciSyncModule* p) {
        if (p && *p) {
            NvSciSyncModuleClose(*p);
        }
        if (p) {
            delete p;
        }
    }
};

struct NvSciBufObjDeleter {
    void operator()(NvSciBufObj* p) {
        if (p && *p) {
            NvSciBufObjFree(*p);
        }
        if (p) {
            delete p;
        }
    }
};

struct NvSciSyncObjDeleter {
    void operator()(NvSciSyncObj* p) {
        if (p && *p) {
            NvSciSyncObjFree(*p);
        }
        if (p) {
            delete p;
        }
    }
};

struct NvSciSyncCpuWaitContextDeleter {
    void operator()(NvSciSyncCpuWaitContext* p) {
        if (p && *p) {
            NvSciSyncCpuWaitContextFree(*p);
        }
        if (p) {
            delete p;
        }
    }
};

NvSciBufAttrListSptr MakeBufAttrListEmpty(NvSciBufModuleSptr buf_module_sptr) {
    NvSciBufAttrList* attr_list_ptr = new NvSciBufAttrList;
    *attr_list_ptr = nullptr;

    NvSciBufAttrListSptr sptr(attr_list_ptr, NvSciBufAttrListDeleter());
    auto sci_err = NvSciBufAttrListCreate(*buf_module_sptr, attr_list_ptr);
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }
    return sptr;
}

NvSciSyncAttrListSptr MakeSyncAttrListEmpty(NvSciSyncModuleSptr sync_module_sptr) {
    NvSciSyncAttrList* attr_list_ptr = new NvSciSyncAttrList;
    *attr_list_ptr = nullptr;

    NvSciSyncAttrListSptr sptr(attr_list_ptr, NvSciSyncAttrListDeleter());
    auto sci_err = NvSciSyncAttrListCreate(*sync_module_sptr, attr_list_ptr);
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }
    return sptr;
}

NvSciBufAttrListSptr MakeBufAttrListForCpu(NvSciBufModuleSptr buf_module_sptr, hozon::netaos::codec::EncodeInitParam& image_param) {

    NvSciBufAttrList* attr_list_ptr = new NvSciBufAttrList;
    *attr_list_ptr = nullptr;

    NvSciBufAttrListSptr sptr(attr_list_ptr, NvSciBufAttrListDeleter());
    auto sci_err = NvSciBufAttrListCreate(*buf_module_sptr, attr_list_ptr);
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }

    uint32_t plane_count = 0;
    NvSciBufType buf_type = NvSciBufType_Image;
    NvSciBufAttrValColorFmt color_format[NV_SCI_BUF_IMAGE_MAX_PLANES];
    NvSciBufAttrValColorStd color_std[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t plane_width[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t plane_height[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    NvSciBufAttrValImageScanType scan_type = NvSciBufScan_ProgressiveType;

    switch (image_param.yuv_type) {
    case hozon::netaos::codec::kYuvType_NV12:
        plane_count = 2;
        color_format[0] = NvSciColor_Y8;
        color_format[1] = NvSciColor_V8U8;
        color_std[0] = NvSciColorStd_REC709_ER;
        color_std[1] = NvSciColorStd_REC709_ER;
        plane_width[0] = image_param.width;
        plane_height[0] = image_param.height;
        plane_width[1] = image_param.width / 2;
        plane_height[1] = image_param.height / 2;
    break;
    case hozon::netaos::codec::kYuvType_YUYV:
        plane_count = 1;
        color_format[0] = NvSciColor_U8Y8V8Y8;
        color_std[0] = NvSciColorStd_REC709_ER;
        plane_width[0] = image_param.width * 2;
        plane_height[0] = image_param.height * 2;
    break;
    case hozon::netaos::codec::kYuvType_YUV420P:
        plane_count = 3;
        color_format[0] = NvSciColor_Y8;
        color_format[1] = NvSciColor_U8;
        color_format[2] = NvSciColor_V8;
        color_std[0] = NvSciColorStd_REC709_ER;
        color_std[1] = NvSciColorStd_REC709_ER;
        color_std[2] = NvSciColorStd_REC709_ER;
        plane_width[0] = image_param.width;
        plane_height[0] = image_param.height;
        plane_width[1] = image_param.width / 4;
        plane_height[1] = image_param.height / 4;
        plane_width[2] = image_param.width / 4;
        plane_height[2] = image_param.height / 4;
    break;
    default:
    return nullptr;
    break;
    }

    bool enable_cpu_cache = true;
    bool need_cpu_access = true;
    NvSciBufAttrKeyValuePair keyvals[] = {
        { NvSciBufGeneralAttrKey_Types, &buf_type, sizeof(buf_type) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &color_format, sizeof(NvSciBufAttrValColorFmt) * plane_count},
        { NvSciBufImageAttrKey_Layout, &image_param.input_mem_layout, sizeof(image_param.input_mem_layout)},
        { NvSciBufImageAttrKey_PlaneCount, &plane_count, sizeof(plane_count)},
        { NvSciBufImageAttrKey_PlaneColorStd, &color_std,sizeof(NvSciBufAttrValColorStd) * plane_count},
        { NvSciBufImageAttrKey_PlaneWidth, &plane_width, plane_count * sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneHeight, &plane_height, plane_count * sizeof(uint32_t) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &need_cpu_access, sizeof(need_cpu_access) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &enable_cpu_cache, sizeof(enable_cpu_cache) },
        { NvSciBufImageAttrKey_ScanType, &scan_type, sizeof(scan_type) }
    };

    sci_err = NvSciBufAttrListSetAttrs(*attr_list_ptr, keyvals, sizeof(keyvals) / sizeof(NvSciBufAttrKeyValuePair));
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }

    return sptr;
}

NvSciSyncAttrListSptr MakeSyncAttrListForCpu(NvSciSyncModuleSptr sync_module_sptr, bool dir) {

    NvSciSyncAttrList* attr_list_ptr = new NvSciSyncAttrList;
    *attr_list_ptr = nullptr;

    NvSciSyncAttrListSptr sptr(attr_list_ptr, NvSciSyncAttrListDeleter());
    auto sci_err = NvSciSyncAttrListCreate(*sync_module_sptr, attr_list_ptr);
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }

    uint8_t cpu_sync = 1;
    NvSciSyncAccessPerm cpu_perm = dir ? NvSciSyncAccessPerm_SignalOnly : NvSciSyncAccessPerm_WaitOnly;

    NvSciSyncAttrKeyValuePair keyvals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpu_sync, sizeof(cpu_sync) },
        { NvSciSyncAttrKey_RequiredPerm,  &cpu_perm, sizeof(cpu_perm) }
    };

    sci_err = NvSciSyncAttrListSetAttrs(*attr_list_ptr, keyvals, sizeof(keyvals) / sizeof(NvSciSyncAttrKeyValuePair));
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }

    return sptr;
}

NvSciBufModuleSptr MakeBufModule() {
    NvSciBufModule* buf_module_ptr = new NvSciBufModule;
    *buf_module_ptr = nullptr;

    NvSciBufModuleSptr sptr(buf_module_ptr, NvSciBufModuleDeleter());
    auto sci_err = NvSciBufModuleOpen(buf_module_ptr);
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }

    return sptr;
}

NvSciSyncModuleSptr MakeSyncModule() {
    NvSciSyncModule* sync_module_ptr = new NvSciSyncModule;
    *sync_module_ptr = nullptr;

    NvSciSyncModuleSptr sptr(sync_module_ptr, NvSciSyncModuleDeleter());
    auto sci_err = NvSciSyncModuleOpen(sync_module_ptr);
    if (sci_err != NvSciError_Success) {
        return nullptr;
    }

    return sptr;
}

NvSciSyncCpuWaitContextSptr MakeCpuWaitContext(NvSciSyncModuleSptr sync_module_sptr) {
    NvSciSyncCpuWaitContext* context_ptr = new NvSciSyncCpuWaitContext;
    *context_ptr = nullptr;

    NvSciSyncCpuWaitContextSptr sptr(context_ptr, NvSciSyncCpuWaitContextDeleter());
    NvSciError sci_err = NvSciSyncCpuWaitContextAlloc(*sync_module_sptr, context_ptr);
    CHECK_SCIERR_RET(sci_err, "NvSciSyncCpuWaitContextAlloc", nullptr);

    return sptr;
}

NvSciBufObjSptr NegotiateBuf(NvSciBufModuleSptr buf_module_sptr, hozon::netaos::codec::Encoder& encoder, hozon::netaos::codec::EncodeInitParam& image_param) {
    NvSciBufAttrListSptr buf_attrs_encoder_required = MakeBufAttrListEmpty(buf_module_sptr);
    void* vp = *buf_attrs_encoder_required;
    auto res = encoder.GetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_BufAttrs, &vp);
    CHECK_CODECERR_RET(res, "Get buf attrs (encoder required)", nullptr);

    NvSciBufAttrListSptr buf_attrs_cpu = MakeBufAttrListForCpu(buf_module_sptr, image_param);
    CHECK_FALSE_RET(buf_attrs_cpu, "Make buf attrs for cpu", nullptr);

    /* Combine and reconcile the attribute lists */
    NvSciBufAttrList old_attr_list[2] = { *buf_attrs_cpu, *buf_attrs_encoder_required };
    NvSciBufAttrList conflicts = nullptr;
    NvSciBufAttrList reconciled = nullptr;
    auto sci_err = NvSciBufAttrListReconcile(old_attr_list, 2, &reconciled, &conflicts);
    CHECK_SCIERR_RET(sci_err, "Reoncile buf attr list", nullptr);

    if (nullptr != conflicts) {
        NvSciBufAttrListFree(conflicts);
    }

    NvSciBufAttrList* attr_list_ptr = new NvSciBufAttrList;
    *attr_list_ptr = reconciled;
    
    NvSciBufAttrListSptr buf_attr_list_sptr(attr_list_ptr, NvSciBufAttrListDeleter());

    res = encoder.SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_BufAttrs, *buf_attr_list_sptr);
    CHECK_CODECERR_RET(res, "Set buf attr list", nullptr);

    NvSciBufObjSptr buf_obj_sptr(new NvSciBufObj, NvSciBufObjDeleter());
    *buf_obj_sptr = nullptr;
    sci_err = NvSciBufObjAlloc(*buf_attr_list_sptr, buf_obj_sptr.get());
    CHECK_SCIERR_RET(sci_err, "Allocate buf object", nullptr);

    res = encoder.SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_BufObj, *buf_obj_sptr);
    CHECK_CODECERR_RET(res, "Set buf object", nullptr);

    return buf_obj_sptr;
}

NvSciSyncObjSptr NegotiateSignalerSync(NvSciSyncModuleSptr sync_module_sptr, hozon::netaos::codec::Encoder& encoder) {
    NvSciSyncAttrListSptr encoder_waitter_attr_list_sptr = MakeSyncAttrListEmpty(sync_module_sptr);
    void* vp = *encoder_waitter_attr_list_sptr;
    auto res = encoder.GetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_WaiterAttrs, &vp);
    CHECK_CODECERR_RET(res, "Get encoder waiter attrs (encoder required)", nullptr);

    NvSciSyncAttrListSptr cpu_signaler_attr_list_sptr = MakeSyncAttrListForCpu(sync_module_sptr, true);
    CHECK_FALSE_RET(cpu_signaler_attr_list_sptr, "Make signaler attr list for cpu", nullptr);

    NvSciSyncAttrList unreconciled[2] = {*cpu_signaler_attr_list_sptr, *encoder_waitter_attr_list_sptr};
    NvSciSyncAttrListSptr reconciled_sptr(new NvSciSyncAttrList, NvSciSyncAttrListDeleter());
    *reconciled_sptr = nullptr;
    NvSciSyncAttrListSptr conflicts_sptr(new NvSciSyncAttrList, NvSciSyncAttrListDeleter());
    *conflicts_sptr = nullptr;

    auto sci_err = NvSciSyncAttrListReconcile(unreconciled, 2, reconciled_sptr.get(), conflicts_sptr.get());
    CHECK_SCIERR_RET(sci_err, "Reconcile signaler (cpu -> encoder) attr list", nullptr);

    NvSciSyncObjSptr signaler_sync_obj_sptr(new NvSciSyncObj, NvSciSyncObjDeleter());
    *signaler_sync_obj_sptr = nullptr;
    sci_err = NvSciSyncObjAlloc(*reconciled_sptr, signaler_sync_obj_sptr.get());
    CHECK_SCIERR_RET(sci_err, "Allocate signaler (cpu -> encoder) sync object", nullptr);

    res = encoder.SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_WaiterObj, *signaler_sync_obj_sptr);
    CHECK_CODECERR_RET(res, "Set encoder waiter sync object", nullptr);

    return signaler_sync_obj_sptr;
}

NvSciSyncObjSptr NegotiateWaiterSync(NvSciSyncModuleSptr sync_module_sptr, hozon::netaos::codec::Encoder& encoder) {
    NvSciSyncAttrListSptr encoder_signaler_attr_list_sptr = MakeSyncAttrListEmpty(sync_module_sptr);
    void* vp = *encoder_signaler_attr_list_sptr;
    auto res = encoder.GetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_SignalerAttrs, &vp);
    CHECK_CODECERR_RET(res, "Get encoder signaler attrs (encoder required)", nullptr);

    NvSciSyncAttrListSptr cpu_waiter_attr_list_sptr = MakeSyncAttrListForCpu(sync_module_sptr, false);
    CHECK_FALSE_RET(cpu_waiter_attr_list_sptr, "Make waiter attr list for cpu", nullptr);

    NvSciSyncAttrList unreconciled[2] = {*cpu_waiter_attr_list_sptr, *encoder_signaler_attr_list_sptr};

    NvSciSyncAttrListSptr reconciled_sptr(new NvSciSyncAttrList, NvSciSyncAttrListDeleter());
    *reconciled_sptr = nullptr;
    NvSciSyncAttrListSptr conflicts_sptr(new NvSciSyncAttrList, NvSciSyncAttrListDeleter());
    *conflicts_sptr = nullptr;

    auto sci_err = NvSciSyncAttrListReconcile(unreconciled, 2, reconciled_sptr.get(), conflicts_sptr.get());
    CHECK_SCIERR_RET(sci_err, "Reconcile waiter (encoder -> cpu) attr list", nullptr);

    NvSciSyncObjSptr waiter_sync_obj_sptr(new NvSciSyncObj, NvSciSyncObjDeleter());
    *waiter_sync_obj_sptr = nullptr;
    sci_err = NvSciSyncObjAlloc(*reconciled_sptr, waiter_sync_obj_sptr.get());
    CHECK_SCIERR_RET(sci_err, "Allocate signaler (cpu -> encoder) sync object", nullptr);

    res = encoder.SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_SignalerObj, *waiter_sync_obj_sptr);
    CHECK_CODECERR_RET(res, "Set encoder signaler sync object", nullptr);

    return waiter_sync_obj_sptr;
}

typedef struct {
    float heightFactor[3];
    float widthFactor[3];
    unsigned int numSurfaces;
} SurfaceDesc;

SurfaceDesc SurfaceDescTable_YUV[hozon::netaos::codec::kYuvType_MAX] = {
        /* kYuvFormat_NV12 */
        {
            /* 420 */
            .heightFactor = {1, 0.5, 0},
            .widthFactor = {1, 0.5, 0},
            .numSurfaces = 2,
        },
        /* kYuvFormat_YUYV */
        {
            /* 422 */
            .heightFactor = {1, 1, 1},
            .widthFactor = {1, 0.5, 0.5},
            .numSurfaces = 3,
        },
        /* kYuvFormat_YUV420P */
        {
            /* 420 */
            .heightFactor = {1, 0.5, 0.5},
            .widthFactor = {1, 0.5, 0.5},
            .numSurfaces = 3,
        }
};

unsigned int SurfaceBytesPerPixelTable_YUV[hozon::netaos::codec::kYuvType_MAX][3] = {
    {1, 2, 0},
    {3, 0, 0},
    {1, 1, 1}
};

int ReadYuvToBufObj(std::vector<uint8_t>& in, uint32_t width, uint32_t height, NvSciBufObj bufObj, uint32_t input_yuv_format) {
    NvSciError err;
    /* Temporary buffer that stores the data read from file */
    uint8_t** pBuff = NULL;
    uint32_t* pBuffPitches = NULL;
    uint8_t* buffer = NULL;
    uint8_t* pBuffer = NULL;
    uint32_t* pBuffSizes = NULL;
    uint32_t imageSize = 0, surfaceSize = 0;
    uint32_t i, j, k, newk = 0;
    /* Number of surfaces in the ChromaFormat of the input file */
    uint32_t numSurfaces = 1;
    uint32_t lumaWidth, lumaHeight;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    FILE* file = NULL;
    unsigned int count, index;
    /* Extract surface info from NvSciBuf */
    NvSciBufAttrList attrList;
    uint32_t const *planeWidth, *planeHeight;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int* srcBytesPerPixelPtr = NULL;
    uint32_t bitsPerPixel = 0U;
    /* Passed outside the function, needs to be static */
    static unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};
    unsigned int numcomps = 1;

    bool uvOrderFlag = true;
    int src_offset = 0;

    if (!bufObj /* || !fileName */) {
        printf("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (input_yuv_format > hozon::netaos::codec::kYuvType_MAX) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Assumptions: This function assumes that
       - Input file is in planar format
       - Input format is of type YUV
       - Output surface is in semi-planar format
       - Output surface can be written to (ops on the surface have completed)
       */

    /* TODO: Add support for other formats */
    xScalePtr = &SurfaceDescTable_YUV[input_yuv_format].widthFactor[0];
    yScalePtr = &SurfaceDescTable_YUV[input_yuv_format].heightFactor[0];
    numSurfaces = SurfaceDescTable_YUV[input_yuv_format].numSurfaces;

    bitsPerPixel = 8U;
    srcBytesPerPixelPtr = &SurfaceBytesPerPixelTable_YUV[input_yuv_format][0];

    if (0U == bitsPerPixel) {
        printf("Failed to deduce bits per pixel");
        return NVMEDIA_STATUS_ERROR;
    }

    err = NvSciBufObjGetAttrList(bufObj, &attrList);
    if (err != NvSciError_Success) {
        printf(" NvSciBufObjGetAttrList failed");
        return NVMEDIA_STATUS_ERROR;
    }

    NvSciBufAttrKeyValuePair imgattrs[] = {
        {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},  /* 0 */
        {NvSciBufImageAttrKey_PlaneHeight, NULL, 0}, /* 1 */
    };
    err = NvSciBufAttrListGetAttrs(attrList, imgattrs, sizeof(imgattrs) / sizeof(NvSciBufAttrKeyValuePair));
    if (err != NvSciError_Success) {
        printf(" NvSciBufAttrListGetAttrs failed");
        return NVMEDIA_STATUS_ERROR;
    }

    planeWidth = (const uint32_t*)(imgattrs[0].value);
    planeHeight = (const uint32_t*)(imgattrs[1].value);
    lumaWidth = planeWidth[0];
    lumaHeight = planeHeight[0];

    /* Check if requested read width, height are lesser than the width and
       height of the surface - checking only for Luma */
    if ((width > lumaWidth) || (height > lumaHeight)) {
        printf("%s: Bad parameter %ux%u vs %ux%u\n", __func__, width, height, lumaWidth, lumaHeight);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuff = reinterpret_cast<uint8_t**>(malloc(sizeof(uint8_t*) * MAX_NUM_SURFACES));
    if (!pBuff) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffSizes = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * MAX_NUM_SURFACES));
    if (!pBuffSizes) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = static_cast<uint32_t*>(calloc(1, sizeof(uint32_t) * MAX_NUM_SURFACES));
    if (!pBuffPitches) {
        printf("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    surfaceSize = 0;
    imageSize = 0;
    for (i = 0; i < numSurfaces; i++) {
        surfaceSize += (lumaWidth * xScalePtr[i] * lumaHeight * yScalePtr[i] * srcBytesPerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * srcBytesPerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)lumaWidth * xScalePtr[i]) * srcBytesPerPixelPtr[i];
    }

    buffer = static_cast<uint8_t*>(calloc(1, surfaceSize));
    if (!buffer) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    memset(buffer, 0x10, surfaceSize);
    for (i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        if (i) {
            memset(pBuff[i], 0x80, (lumaHeight * yScalePtr[i] * pBuffPitches[i]));
        }
        pBuffSizes[i] = (uint32_t)(lumaHeight * yScalePtr[i] * pBuffPitches[i]);
        buffer = buffer + pBuffSizes[i];
    }

    // file = fopen(fileName, "rb");
    // if(!file) {
    //     printf("%s: Error opening file: %s\n", __func__, fileName);
    //     status = NVMEDIA_STATUS_ERROR;
    //     goto done;
    // }

    // if(frameNum > 0) {
    //     if(fseeko(file, frameNum * (off_t)imageSize, SEEK_SET)) {
    //         printf("ReadInput: Error seeking file: %s\n", fileName);
    //         status = NVMEDIA_STATUS_ERROR;
    //         goto done;
    //     }
    // }

    // if (inputFileChromaFormat == YUV420SP_8bit) {  // Assuming NV12
    //     int y_size = pBuffSizes[0];
    //     int u_size = pBuffSizes[1];
    //     int v_size = pBuffSizes[2];

    //     memcpy(pBuff[0], buf, y_size);

    //     int u_dst_offset = 0;
    //     int v_dst_offset = 0;
    //     for (int i = y_size; i < imageSize; i += 2) {
    //         pBuff[1][u_dst_offset++] = buf[i];
    //         pBuff[2][v_dst_offset++] = buf[i + 1];
    //     }

    //     // memcpy(pBuff[0], buf, y_size);
    //     // memcpy(pBuff[1], buf + y_size, u_size + v_size);
    // } else {

        for (k = 0; k < numSurfaces; k++) {
            for (j = 0; j < height * yScalePtr[k]; j++) {
                newk = (!uvOrderFlag && k) ? (numSurfaces - k) : k;
                index = j * pBuffPitches[newk];
                count = width * xScalePtr[newk] * srcBytesPerPixelPtr[newk];
                // if (fread(pBuff[newk] + index, count, 1, file) != 1) {
                memcpy(pBuff[newk] + index, in.data() + src_offset, count * 1);
                src_offset += count * 1;

                /* TODO: Assuming YUV input */
                // if (pixelAlignment == LSB_ALIGNED) {
                    uint16_t* psrc = (uint16_t*)(pBuff[newk] + index);
                    if (bitsPerPixel > 8U) {
                        for (i = 0; i < count / 2; i++) {
                            *(psrc + i) = (*(psrc + i)) << (16 - bitsPerPixel);
                        }
                    }
                // }
            }
        }
    // }

    err = NvSciBufObjPutPixels(bufObj, NULL, (const void**)pBuff, pBuffSizes, pBuffPitches);
    if (err != NvSciError_Success) {
        printf("NvSciBufObjPutPixels failed.");

        status = NVMEDIA_STATUS_ERROR;
        goto done;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

done:
    if (pBuff) {
        free(pBuff);
    }

    if (pBuffSizes) {
        free(pBuffSizes);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if (file) {
        fclose(file);
    }

    return status;
}

static bool stopped_ = false;

/** Signal handler.*/
static void SigHandler(int signum)
{

    std::cout << "Received signal: " << signum  << ". Quitting\n";
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    stopped_ = true;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
static void SigSetup(void)
{
    struct sigaction action
    {
    };
    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGQUIT, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);
}

int main(int argc, const char* argv[]) {

    SigSetup();

    InitializeLogging();

    if (argc != 4) {
        logger_->LogError() << "file_path width height must be specified.";
        return -1;
    }

    std::string file_path(argv[1]);
    uint32_t width = std::stoul(argv[2]);
    uint32_t height = std::stoul(argv[3]);

    logger_->LogInfo() << "encoder_nvmedia_test start.";
    
    auto encoder_uptr = hozon::netaos::codec::EncoderFactory::Create(hozon::netaos::codec::kDeviceType_NvMedia);
    std::string cfg_file;

    hozon::netaos::codec::EncodeInitParam init_param{0};
    init_param.width = width;
    init_param.height = height;
    init_param.codec_type = hozon::netaos::codec::kCodecType_H265;
    init_param.yuv_type = hozon::netaos::codec::kYuvType_NV12;
    init_param.input_buf_type = hozon::netaos::codec::kBufType_SciBuf;
    init_param.input_mem_layout = hozon::netaos::codec::kMemLayout_BL;
    hozon::netaos::codec::CodecErrc res = encoder_uptr->Init(init_param);

    CHECK_CODECERR_RET(res, "Init encoder", -1);

    // Create buf module and sync module and set them to encoder.
    NvSciBufModuleSptr buf_module_sptr = MakeBufModule();
    if (!buf_module_sptr) {
        logger_->LogError() << "Make buf module failed.";
        return -1;
    }
    res = encoder_uptr->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_BufModule, *buf_module_sptr);
    if (res != hozon::netaos::codec::kEncodeSuccess) {
        logger_->LogError() << "Set buf module failed.";
        return -1;
    }

    NvSciSyncModuleSptr sync_module_sptr = MakeSyncModule();
    if (!sync_module_sptr) {
        logger_->LogError() << "Make sync module failed.";
        return -1;
    }
    res = encoder_uptr->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_SyncModule, *sync_module_sptr);
    if (res != hozon::netaos::codec::kEncodeSuccess) {
        logger_->LogError() << "Set buf sync failed.";
        return -1;
    }

    // Negotiate buf with encoder.
    NvSciBufObjSptr buf_obj_sptr = NegotiateBuf(buf_module_sptr, *encoder_uptr, init_param);
    if (!buf_obj_sptr) {
        logger_->LogError() << "Negotiate buf failed.";
        return -1;
    }

    // Negotiate signaler sync (cpu -> encoder) with encoder.
    NvSciSyncObjSptr cpu_signaler_sync_obj_sptr = NegotiateSignalerSync(sync_module_sptr, *encoder_uptr);
    if (!cpu_signaler_sync_obj_sptr) {
        logger_->LogError() << "Negotiate signaler (cpu -> encoder) attr list failed.";
        return -1;
    }

    // Negotiate waiter sync (encoder -> cpu) with encoder.
    NvSciSyncObjSptr cpu_waiter_sync_obj_sptr = NegotiateWaiterSync(sync_module_sptr, *encoder_uptr);
    CHECK_FALSE_RET(cpu_waiter_sync_obj_sptr, "Negotiate waiter (encoder -> cpu) attr list", -1);

    NvSciSyncCpuWaitContextSptr cpu_wait_context_sptr = MakeCpuWaitContext(sync_module_sptr);
    CHECK_FALSE_RET(cpu_wait_context_sptr, "Make cpu wait context", -1);

    // Tell encoder negotiation process is finished.
    res = encoder_uptr->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_SetupComplete, nullptr);
    CHECK_CODECERR_RET(res, "Set setup completed state to encoder", -1);

    // Construct buf object content and encode.
    std::vector<uint8_t> yuv_cpu_buf;
    {
        if (std::ifstream ifs{file_path, std::ios::binary | std::ios::ate}) {
            auto file_size = ifs.tellg();
            uint32_t frame_size = (width * height) * 3 / 2; // test for yuv420p

            if (file_size < frame_size) {
                logger_->LogError() << "Yuv file size is less than yuv frame size.";
                return -1;
            }

            ifs.seekg(0);
            yuv_cpu_buf.resize(frame_size);
            if (!ifs.read(reinterpret_cast<char*>(yuv_cpu_buf.data()), yuv_cpu_buf.size())) {
                logger_->LogError() << "Read yuv file failed.";
                return -1;
            }
        }
    }

    if (yuv_cpu_buf.size() <= 0) {
        logger_->LogError() << "Read yuv_cpu_buf size is not correct: " << yuv_cpu_buf.size();
        return -1;
    }

    int read_res = ReadYuvToBufObj(yuv_cpu_buf, width, height, *buf_obj_sptr, hozon::netaos::codec::kYuvType_YUV420P);
    CHECK_FALSE_RET((read_res == 0), "ReadYuvToBufObj", -1);

    while (!stopped_) {
        NvSciSyncFence prefence = NvSciSyncFenceInitializer;
        NvSciSyncFence eoffence = NvSciSyncFenceInitializer;

        auto sci_err = NvSciSyncObjGenerateFence(*cpu_signaler_sync_obj_sptr, &prefence);
        CHECK_SCIERR_RET(sci_err, "NvSciSyncObjGenerateFence", -1);
        sci_err = NvSciSyncObjSignal(*cpu_signaler_sync_obj_sptr);
        CHECK_SCIERR_RET(sci_err, "NvSciSyncObjSignal", -1);

        hozon::netaos::codec::EncoderBufNvSpecific buf_nv;
        buf_nv.buf_obj = *buf_obj_sptr;
        buf_nv.pre_fence = &prefence;
        buf_nv.eof_fence = &eoffence;
        std::vector<uint8_t> buf265;
        hozon::netaos::codec::FrameType frame_type;
        res = encoder_uptr->Process(&buf_nv, buf265, frame_type);
        CHECK_CODECERR_RET(res, "Encode", -1);

        sci_err = NvSciSyncFenceWait(&eoffence, *cpu_wait_context_sptr, 1000 * 1000);
        CHECK_SCIERR_RET(sci_err, "NvSciSyncFenceWait", -1);

        std::ofstream ofs("encoder_nvmedia_test.265", std::ios::binary | std::ios::app | std::ios::out);
        ofs.write(reinterpret_cast<char*>(buf265.data()), buf265.size());

        logger_->LogInfo() << "Encode success. h265 size: " << buf265.size();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    logger_->LogInfo() << "encoder_nvmedia_test end.";


    return 0;
}