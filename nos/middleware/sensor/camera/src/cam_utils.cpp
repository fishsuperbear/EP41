/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Standard header files
#include <cstring>

// Sample application header files
#include "cam_filewriter.hpp"
#include "cam_utils.hpp"


namespace hozon {
namespace netaos {
namespace camera {

SIPLStatus LoadNitoFile(std::string const &folderPath,
                        std::string const &moduleName,
                        std::vector<uint8_t> &nito,
                        bool &defaultLoaded)
{
    CFileManager moduleNito(folderPath + moduleName + ".nito", "rb");
    CFileManager defaultNito(folderPath + "default.nito", "rb");
    FILE *fp = nullptr;
    defaultLoaded = false;

    if (moduleNito.GetFile() != nullptr) {
        CAM_LOG_INFO << "Opened NITO file for module : " << moduleName.c_str();
        fp = moduleNito.GetFile();
    } else {
        CAM_LOG_ERROR << "Unable to open NITO file for module " << moduleName.c_str();
        if (defaultNito.GetFile() != nullptr) {
            CAM_LOG_INFO << "Opened default NITO file for module " << defaultNito.GetName().c_str();
            fp = defaultNito.GetFile();
            defaultLoaded = true;
        } else {
            CAM_LOG_ERROR << "Unable to open default NITO file " << defaultNito.GetName().c_str();
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

    // Check file size
    fseek(fp, 0, SEEK_END);
    size_t fsize = ftell(fp);
    rewind(fp);

    if (fsize <= 0U) {
        CAM_LOG_ERROR << "NITO file for module is of invalid size " << moduleName.c_str();
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Allocate blob memory
    nito.resize(fsize);

    // Load NITO
    size_t result = fread(nito.data(), 1U, fsize, fp);
    if (result != fsize) {
        CAM_LOG_ERROR << "Unable to read data from NITO file for module, image quality is not supported  "
                << moduleName.c_str();
        nito.resize(0U);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    CAM_LOG_INFO << "Data from NITO file loaded for module ", moduleName.c_str();

    return NVSIPL_STATUS_OK;
}

SIPLStatus GetEventName(const NvSIPLPipelineNotifier::NotificationData &event, const char *&eventName)
{
    static const EventMap eventNameTable[] = {
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ICP_PROCESSING_DONE,
            "NOTIF_INFO_ICP_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ISP_PROCESSING_DONE,
            "NOTIF_INFO_ISP_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ACP_PROCESSING_DONE,
            "NOTIF_INFO_ACP_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_ICP_AUTH_SUCCESS,
            "NOTIF_INFO_ICP_AUTH_SUCCESS"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_INFO_CDI_PROCESSING_DONE,
            "NOTIF_INFO_CDI_PROCESSING_DONE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DROP,
            "NOTIF_WARN_ICP_FRAME_DROP"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DISCONTINUITY,
            "NOTIF_WARN_ICP_FRAME_DISCONTINUITY"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_CAPTURE_TIMEOUT,
            "NOTIF_WARN_ICP_CAPTURE_TIMEOUT"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_BAD_INPUT_STREAM,
            "NOTIF_ERROR_ICP_BAD_INPUT_STREAM"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_CAPTURE_FAILURE,
            "NOTIF_ERROR_ICP_CAPTURE_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE,
            "NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ISP_PROCESSING_FAILURE,
            "NOTIF_ERROR_ISP_PROCESSING_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ACP_PROCESSING_FAILURE,
            "NOTIF_ERROR_ACP_PROCESSING_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE,
            "NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_INTERNAL_FAILURE,
            "NOTIF_ERROR_INTERNAL_FAILURE"},
        {NvSIPLPipelineNotifier::NotificationType::NOTIF_ERROR_ICP_AUTH_FAILURE,
            "NOTIF_ERROR_ICP_AUTH_FAILURE"}
    };

    for (uint32_t i = 0U; i < ARRAY_SIZE(eventNameTable); i++) {
        if (event.eNotifType == eventNameTable[i].eventType) {
            eventName = eventNameTable[i].eventName;
            return NVSIPL_STATUS_OK;
        }
    }

    CAM_LOG_ERROR << "Unknown event type.";
    return NVSIPL_STATUS_BAD_ARGUMENT;
}

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs &bufAttrs)
{
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttrList;

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        { NvSciBufImageAttrKey_Size, NULL, 0 },               //0
        { NvSciBufImageAttrKey_Layout, NULL, 0 },             //1
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },         //2
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },         //3
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },        //4
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },         //5
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },  //6
        { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 }, //7
        { NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0 },   //8
        { NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0 },  //9
        { NvSciBufImageAttrKey_PlaneOffset, NULL, 0 },        //10
        { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },    //11
        { NvSciBufImageAttrKey_TopPadding, NULL, 0 },        //12
        { NvSciBufImageAttrKey_BottomPadding, NULL, 0 },   //13
        { NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0 } //14
    };

    err = NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjGetAttrList");
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, sizeof(imgAttrs) / sizeof(imgAttrs[0]));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListGetAttrs");

    bufAttrs.size = *(static_cast<const uint64_t*>(imgAttrs[0].value));
    bufAttrs.layout = *(static_cast<const NvSciBufAttrValImageLayoutType*>(imgAttrs[1].value));
    bufAttrs.planeCount = *(static_cast<const uint32_t*>(imgAttrs[2].value));
    bufAttrs.needSwCacheCoherency = *(static_cast<const bool*>(imgAttrs[14].value));

    memcpy(bufAttrs.planeWidths,
        static_cast<const uint32_t*>(imgAttrs[3].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeHeights,
        static_cast<const uint32_t*>(imgAttrs[4].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planePitches,
        static_cast<const uint32_t*>(imgAttrs[5].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeBitsPerPixels,
        static_cast<const uint32_t*>(imgAttrs[6].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedHeights,
        static_cast<const uint32_t*>(imgAttrs[7].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedSizes,
        static_cast<const uint64_t*>(imgAttrs[8].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeChannelCounts,
        static_cast<const uint8_t*>(imgAttrs[9].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeOffsets,
        static_cast<const uint64_t*>(imgAttrs[10].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeColorFormats,
        static_cast<const NvSciBufAttrValColorFmt*>(imgAttrs[11].value),
        bufAttrs.planeCount * sizeof(NvSciBufAttrValColorFmt));
    memcpy(bufAttrs.topPadding,
        static_cast<const uint32_t*>(imgAttrs[12].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.bottomPadding,
        static_cast<const uint32_t*>(imgAttrs[13].value),
        bufAttrs.planeCount * sizeof(uint32_t));

    return NVSIPL_STATUS_OK;
}

SIPLStatus CUtils::IsRawBuffer(NvSciBufObj bufObj, bool &bIsRaw)
{
    bIsRaw = false;
    BufferAttrs bufAttrs;
    SIPLStatus status = PopulateBufAttr(bufObj, bufAttrs);
    CHK_STATUS_AND_RETURN(status, "PopulateBufAttr");
    NvSciBufAttrValColorFmt colorFmt = bufAttrs.planeColorFormats[0];
    if (((colorFmt >= NvSciColor_Bayer8RGGB) && (colorFmt <= NvSciColor_Signed_X12Bayer20GBRG)) ||
        (colorFmt == NvSciColor_X4Bayer12RGGB_RJ)) {
        bIsRaw = true;
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CUtils::CreateRgbaBuffer(NvSciBufModule &bufModule,
                                    NvSciBufAttrList &bufAttrList,
                                    uint32_t width,
                                    uint32_t height,
                                    NvSciBufObj *pBufObj)
{
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    bool imgCpuAccess = true;
    bool imgCpuCacheEnabled = true;
    uint32_t planeCount = 1U;
    NvSciBufAttrValColorFmt planeColorFmt = NvSciColor_A8B8G8R8;
    NvSciBufAttrValColorStd planeColorStd = NvSciColorStd_SRGB;
    NvSciBufAttrValImageLayoutType imgLayout = NvSciBufImage_PitchLinearType;
    uint64_t zeroPadding = 0U;
    uint32_t planeWidth = width;
    uint32_t planeHeight = height;
    uint32_t planeBaseAddrAlign = 256U;

    NvSciBufAttrKeyValuePair setAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(NvSciBufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(NvSciBufAttrValAccessPerm) },
        { NvSciBufImageAttrKey_ScanType, &bufScanType, sizeof(bufScanType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &imgCpuAccess, sizeof(bool) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &imgCpuCacheEnabled, sizeof(bool) },
        { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &planeColorFmt, sizeof(NvSciBufAttrValColorFmt) },
        { NvSciBufImageAttrKey_PlaneColorStd, &planeColorStd, sizeof(NvSciBufAttrValColorStd) },
        { NvSciBufImageAttrKey_Layout, &imgLayout, sizeof(NvSciBufAttrValImageLayoutType) },
        { NvSciBufImageAttrKey_TopPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_BottomPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_LeftPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_RightPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_PlaneWidth, &planeWidth, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneHeight, &planeHeight, sizeof(uint32_t)  },
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &planeBaseAddrAlign, sizeof(uint32_t) }
    };
    size_t length = sizeof(setAttrs) / sizeof(NvSciBufAttrKeyValuePair);

    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> unreconciledAttrList;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> reconciledAttrList;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> conflictAttrList;
    unreconciledAttrList.reset(new NvSciBufAttrList());
    reconciledAttrList.reset(new NvSciBufAttrList());
    conflictAttrList.reset(new NvSciBufAttrList());
    NvSciError sciErr = NvSciBufAttrListCreate(bufModule, unreconciledAttrList.get());
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");
    sciErr = NvSciBufAttrListSetAttrs(*unreconciledAttrList, setAttrs, length);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    NvSciBufAttrList unreconciledAttrLists[2] = { *unreconciledAttrList, bufAttrList };
    sciErr = NvSciBufAttrListReconcile(unreconciledAttrLists,
                                       2U,
                                       reconciledAttrList.get(),
                                       conflictAttrList.get());
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListReconcile");
    sciErr = NvSciBufObjAlloc(*reconciledAttrList, pBufObj);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjAlloc");
    CHK_PTR_AND_RETURN(pBufObj, "NvSciBufObjAlloc");

    return NVSIPL_STATUS_OK;
}

uint8_t * CUtils::CreateImageBuffer(NvSciBufObj bufObj)
{
    BufferAttrs bufAttrs;
    SIPLStatus status = PopulateBufAttr(bufObj, bufAttrs);
    if (status != NVSIPL_STATUS_OK) {
        CAM_LOG_ERROR << "PopulateBufAttr failed. status: " << status;
        return nullptr;
    }
    uint8_t *buff = new (std::nothrow) uint8_t[bufAttrs.size];
    if (buff == nullptr) {
        CAM_LOG_ERROR << "Failed to allocate memory for image buffer.";
        return nullptr;
    }
    std::fill(buff, buff + bufAttrs.size, 0x00);

    return buff;
}

bool CUtils::GetBpp(uint32_t buffBits, uint32_t *buffBytesVal) {
    uint32_t buffBytes = 0U;
    if (buffBytesVal == NULL) {
        return false;
    }
    switch(buffBits) {
        case 8:
            buffBytes = 1U;
            break;
        case 10:
        case 12:
        case 14:
        case 16:
            buffBytes = 2U;
            break;
        case 20:
            buffBytes = 3U;
            break;
        case 32:
            buffBytes = 4U;
            break;
        case 64:
            buffBytes = 8U;
            break;
        default:
            CAM_LOG_ERROR << "Invalid planeBitsPerPixels : " << buffBits;
            return false;
    }
    *buffBytesVal = buffBytes;
    return true;
}

static inline uint8_t getPixel16BitsLE(const uint8_t *pPixBuf)
{
    uint16_t pix = ((uint16_t)pPixBuf[0]) | ((uint16_t)pPixBuf[1] << 8);

    return (uint8_t)(pix >> 4);
}

SIPLStatus CUtils::ConvertRawToRgba(NvSciBufObj srcBufObj,
                                    uint8_t *pSrcBuf,
                                    NvSciBufObj dstBufObj,
                                    uint8_t *pDstBuf)
{
    BufferAttrs srcBufAttrs;
    SIPLStatus status = PopulateBufAttr(srcBufObj, srcBufAttrs);
    CHK_STATUS_AND_RETURN(status, "PopulateBufAttr for source buffer");
    uint8_t *pSrcBufCpy = pSrcBuf;
    uint8_t *pDstBufCpy = pDstBuf;
    bool pixIsRJ = false;

    NvSciError sciErr = NvSciError_Success;
    if (srcBufAttrs.needSwCacheCoherency) {
        sciErr = NvSciBufObjFlushCpuCacheRange(srcBufObj,
                                               0U,
                                               srcBufAttrs.planePitches[0]
                                                   * srcBufAttrs.planeHeights[0]);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjFlushCpuCacheRange");
    }

    uint32_t bpp = 0U;
    if (!GetBpp(srcBufAttrs.planeBitsPerPixels[0], &bpp)) {
        return NVSIPL_STATUS_ERROR;
    }

    const uint32_t srcPitch = srcBufAttrs.planeWidths[0] * bpp;
    const uint32_t srcBufSize = srcPitch * srcBufAttrs.planeHeights[0];
    sciErr = NvSciBufObjGetPixels(srcBufObj,
                                  nullptr,
                                  (void **)(&pSrcBufCpy),
                                  &srcBufSize,
                                  &srcPitch);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetPixels");
    // Do CPU demosaic
    // Get offsets for each raw component within 2x2 block
    uint32_t xR = 0U, yR = 0U, xG1 = 0U, yG1 = 0U, xG2 = 0U, yG2 = 0U, xB = 0U, yB = 0U;
    switch (srcBufAttrs.planeColorFormats[0]) {
        case NvSciColor_Bayer8RGGB:
        case NvSciColor_Bayer16RGGB:
        case NvSciColor_X2Bayer14RGGB:
        case NvSciColor_X4Bayer12RGGB:
        case NvSciColor_X4Bayer12RGGB_RJ:
        case NvSciColor_X6Bayer10RGGB:
        case NvSciColor_FloatISP_Bayer16RGGB:
        case NvSciColor_X12Bayer20RGGB:
        case NvSciColor_Bayer16RCCB:
        case NvSciColor_X4Bayer12RCCB:
        case NvSciColor_FloatISP_Bayer16RCCB:
        case NvSciColor_X12Bayer20RCCB:
        case NvSciColor_Bayer16RCCC:
        case NvSciColor_X4Bayer12RCCC:
        case NvSciColor_FloatISP_Bayer16RCCC:
        case NvSciColor_X12Bayer20RCCC:
            xR = 0U; yR = 0U;
            xG1 = 1U; yG1 = 0U;
            xG2 = 0U; yG2 = 1U;
            xB = 1U; yB = 1U;
            break;
        case NvSciColor_Bayer8GRBG:
        case NvSciColor_Bayer16GRBG:
        case NvSciColor_X2Bayer14GRBG:
        case NvSciColor_X4Bayer12GRBG:
        case NvSciColor_X6Bayer10GRBG:
        case NvSciColor_FloatISP_Bayer16GRBG:
        case NvSciColor_X12Bayer20GRBG:
        case NvSciColor_Bayer16CRBC:
        case NvSciColor_X4Bayer12CRBC:
        case NvSciColor_FloatISP_Bayer16CRBC:
        case NvSciColor_X12Bayer20CRBC:
        case NvSciColor_Bayer16CRCC:
        case NvSciColor_X4Bayer12CRCC:
        case NvSciColor_FloatISP_Bayer16CRCC:
        case NvSciColor_X12Bayer20CRCC:
            xG1 = 0U; yG1 = 0U;
            xR = 1U; yR = 0U;
            xB = 0U; yB = 1U;
            xG2 = 1U; yG2 = 1U;
            break;
        case NvSciColor_Bayer8GBRG:
        case NvSciColor_Bayer16GBRG:
        case NvSciColor_X2Bayer14GBRG:
        case NvSciColor_X4Bayer12GBRG:
        case NvSciColor_X6Bayer10GBRG:
        case NvSciColor_FloatISP_Bayer16GBRG:
        case NvSciColor_X12Bayer20GBRG:
        case NvSciColor_Signed_X12Bayer20GBRG:
        case NvSciColor_Bayer16CBRC:
        case NvSciColor_X4Bayer12CBRC:
        case NvSciColor_FloatISP_Bayer16CBRC:
        case NvSciColor_X12Bayer20CBRC:
        case NvSciColor_Bayer16CCRC:
        case NvSciColor_X4Bayer12CCRC:
        case NvSciColor_FloatISP_Bayer16CCRC:
        case NvSciColor_X12Bayer20CCRC:
            xG1 = 0U; yG1 = 0U;
            xB = 1U; yB = 0U;
            xR = 0U; yR = 1U;
            xG2 = 1U; yG2 = 1U;
            break;
        case NvSciColor_Bayer8BGGR:
        case NvSciColor_Bayer16BGGR:
        case NvSciColor_X2Bayer14BGGR:
        case NvSciColor_X4Bayer12BGGR:
        case NvSciColor_X6Bayer10BGGR:
        case NvSciColor_FloatISP_Bayer16BGGR:
        case NvSciColor_X12Bayer20BGGR:
        case NvSciColor_Bayer16BCCR:
        case NvSciColor_X4Bayer12BCCR:
        case NvSciColor_FloatISP_Bayer16BCCR:
        case NvSciColor_X12Bayer20BCCR:
        case NvSciColor_Bayer16CCCR:
        case NvSciColor_X4Bayer12CCCR:
        case NvSciColor_FloatISP_Bayer16CCCR:
        case NvSciColor_X12Bayer20CCCR:
        case NvSciColor_Bayer8CCCC:
        case NvSciColor_Bayer16CCCC:
        case NvSciColor_X2Bayer14CCCC:
        case NvSciColor_X4Bayer12CCCC:
        case NvSciColor_X6Bayer10CCCC:
        case NvSciColor_Signed_X2Bayer14CCCC:
        case NvSciColor_Signed_X4Bayer12CCCC:
        case NvSciColor_Signed_X6Bayer10CCCC:
        case NvSciColor_Signed_Bayer16CCCC:
        case NvSciColor_FloatISP_Bayer16CCCC:
        case NvSciColor_X12Bayer20CCCC:
        case NvSciColor_Signed_X12Bayer20CCCC:
            xB = 0U; yB = 0U;
            xG1 = 1U; yG1 = 0U;
            xG2 = 0U; yG2 = 1U;
            xR = 1U; yR = 1U;
            break;
        default:
            CAM_LOG_ERROR << "Unexpected plane color format";
            return NVSIPL_STATUS_ERROR;
    }

    if (srcBufAttrs.planeColorFormats[0] == NvSciColor_X4Bayer12RGGB_RJ) {
        /* Right justified format (pixel starts with least significant bit) */
        pixIsRJ = true;
    }

    // Demosaic, remembering to skip embedded lines
    for (uint32_t y = srcBufAttrs.topPadding[0];
         y < (srcBufAttrs.planeHeights[0] - static_cast<uint32_t>(srcBufAttrs.bottomPadding[0]));
         y += 2U) {
        for (uint32_t x = 0U; x < srcBufAttrs.planeWidths[0]; x += 2U) {
            if (pixIsRJ) {
                // R
                *pDstBuf++ = getPixel16BitsLE(&pSrcBuf[srcPitch*(y + yR) + 2U*(x + xR)]);
                // G (average of G1 and G2)
                uint32_t g1 = getPixel16BitsLE(&pSrcBuf[srcPitch*(y + yG1) + 2U*(x + xG1)]);
                uint32_t g2 = getPixel16BitsLE(&pSrcBuf[srcPitch*(y + yG2) + 2U*(x + xG2)]);
                *pDstBuf++ = (g1 + g2)/2U;
                // B
                *pDstBuf++ = getPixel16BitsLE(&pSrcBuf[srcPitch*(y + yB) + 2U*(x + xB)]);
            } else {
                // R
                *pDstBuf++ = pSrcBuf[srcPitch*(y + yR) + 2U*(x + xR) + 1U];
                // G (average of G1 and G2)
                uint32_t g1 = pSrcBuf[srcPitch*(y + yG1) + 2U*(x + xG1) + 1U];
                uint32_t g2 = pSrcBuf[srcPitch*(y + yG2) + 2U*(x + xG2) + 1U];
                *pDstBuf++ = (g1 + g2)/2U;
                // B
                *pDstBuf++ = pSrcBuf[srcPitch*(y + yB) + 2U*(x + xB) + 1U];
            }
            // A
            *pDstBuf++ = 0xFF;
        }
    }

    // Write to destination image
    BufferAttrs dstBufAttrs;
    status = PopulateBufAttr(dstBufObj, dstBufAttrs);
    CHK_STATUS_AND_RETURN(status, "PopulateBufAttr for destination buffer");

    if (!GetBpp(dstBufAttrs.planeBitsPerPixels[0], &bpp)) {
        return NVSIPL_STATUS_ERROR;
    }
    const uint32_t dstPitch = dstBufAttrs.planeWidths[0] * bpp;
    const uint32_t dstBufSize = dstPitch * dstBufAttrs.planeHeights[0];
    sciErr = NvSciBufObjPutPixels(dstBufObj,
                                  nullptr,
                                  (const void **)(&pDstBufCpy),
                                  &dstBufSize,
                                  &dstPitch);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjPutPixels");

    return NVSIPL_STATUS_OK;
}

SIPLStatus WriteImageToFile(INvSIPLClient::INvSIPLNvMBuffer * pNvMBuffer, uint32_t uSensorId, uint32_t uFrameCount) 
{
    std::string sFileExt;
    std::unique_ptr<CFileWriter> m_pFileWriter = std::make_unique<CFileWriter>();
    NvSciBufObj bufPtr = pNvMBuffer->GetNvSciBufImage();
    BufferAttrs bufAttrs;
    auto status = PopulateBufAttr(bufPtr, bufAttrs);
    if(status != NVSIPL_STATUS_OK) {
        CAM_LOG_ERROR << "Consumer: PopulateBufAttr failed. ";
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    if (((bufAttrs.planeColorFormats[0] > NvSciColor_LowerBound) &&
            (bufAttrs.planeColorFormats[0] < NvSciColor_U8V8)) ||
        ((bufAttrs.planeColorFormats[0] > NvSciColor_Float_A16) &&
            (bufAttrs.planeColorFormats[0] < NvSciColor_UpperBound))) {
        sFileExt = ".raw";
    } else if ((bufAttrs.planeColorFormats[0] == NvSciColor_Y16) && (bufAttrs.planeCount == 1U)) {
        sFileExt = ".luma";
    } else if (((bufAttrs.planeColorFormats[0] > NvSciColor_V16U16) &&
                (bufAttrs.planeColorFormats[0] < NvSciColor_U8)) ||
                ((bufAttrs.planeColorFormats[0] > NvSciColor_V16) &&
                (bufAttrs.planeColorFormats[0] < NvSciColor_A8))) {
        sFileExt = ".yuv";
    } else if ((bufAttrs.planeColorFormats[0] > NvSciColor_A16Y16U16V16) &&
                (bufAttrs.planeColorFormats[0] < NvSciColor_X6Bayer10BGGI_RGGI)) {
        sFileExt = ".rgba";
    }

    std::string sFilename = "cam_" + std::to_string(uSensorId)
                            + "_out_" + std::to_string(uFrameCount) + sFileExt;
    bool bRawOut = true;
    status = m_pFileWriter->Init(sFilename, bRawOut);
    if (status != NVSIPL_STATUS_OK) {
        CAM_LOG_ERROR << "Failed to initialize file writer.";
        return status;
    }
    status = m_pFileWriter->WriteBufferToFile(pNvMBuffer);
    if (status != NVSIPL_STATUS_OK) {
        CAM_LOG_ERROR << "WriteBufferToFile failed";
        return status;
    }

    return NVSIPL_STATUS_OK;
}

void YUYV2NV12(uint32_t width, uint32_t height, const std::string& yuyv, std::string& nv12) {
    if (width*height*2 != yuyv.size()) {
        CAM_LOG_ERROR << "Invalid yuyv size " << yuyv.size() << ", should be " << width*height*2;
        return;
    }

    nv12.resize(width*height*3/2);

    // y
    uint32_t y_size = width * height;
    for (uint32_t i = 0; i < y_size; ++i) {
        nv12[i] = yuyv[i*2];

    }

    // uv
    uint32_t uv_begin = width * height;
    uint32_t uv_size = width * height / 4;
    for (uint32_t i = 0; i < uv_size; ++i) {
        uint32_t line = i / (width / 2);
        nv12[2*i+uv_begin] = yuyv[4*i + 1 + 2*width*line];
        nv12[2*i+uv_begin+1] = yuyv[4*i + 3 + 2*width*line];
    }
}

void YUY4202NV12(uint32_t width, uint32_t height, const std::string& yuv, std::string& nv12) {
    if (width*height*3/2 != yuv.size()) {
        CAM_LOG_ERROR << "Invalid yuv size " << yuv.size() << ", should be " << width*height*2;
        return;
    }

    nv12.resize(width*height*3/2);

    // y
    uint32_t y_size = width * height;
    for (uint32_t i = 0; i < y_size; ++i) {
        nv12[i] = yuv[i];
    }

    // uv
    uint32_t u_begin = width * height;
    uint32_t v_begin = width * height + width * height / 4;
    uint32_t uv_size = width * height / 4;
    for (uint32_t i = 0; i < uv_size; ++i) {
        nv12[u_begin + 2 * i] = yuv[u_begin + i];
        nv12[u_begin + 2 * i + 1] = yuv[v_begin + i];
    }
}

}
}
}
