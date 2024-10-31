// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "sensor/nvs_consumer/CNvMediaConsumer.hpp"
#include "nvmedia_2d_sci.h"
#include "sensor/nvs_consumer/CUtils.hpp"
#include "cuda_runtime_api.h"
#include "cuda.h"

namespace hozon {
namespace netaos {
namespace desay { 

// #include "nvmedia_iep_nvscisync.h"

CNvMediaConsumer::CNvMediaConsumer(NvSciStreamBlock handle,
                               uint32_t uSensor,
                               NvSciStreamBlock queueHandle,
                               uint16_t NvMediaWidth,
                               uint16_t NvMediaHeight) :
    CConsumer("MediaConsumer", handle, uSensor, queueHandle)
{
    m_Width = NvMediaWidth;
    m_Height = NvMediaHeight;
}

SIPLStatus CNvMediaConsumer::HandleClientInit(void)
{
    NvMediaStatus mediaErr = NvMedia2DCreate(&m_pNvMedia,NULL);
    PCHK_NVMSTATUS_AND_RETURN(mediaErr, "NvMedia2DCreate");
    auto status = CreateImage();
    PCHK_STATUS_AND_RETURN(status, "CreateImage");
    string timeFileName = "/tmp/multicast_img" + to_string(m_uSensorId) + ".time";
    m_pOutputTimeFile = fopen(timeFileName.c_str(), "wb");
    string imgFileName = "/tmp/multicast_img" + to_string(m_uSensorId) + ".rgba";
    m_pOutputImgFile = fopen(imgFileName.c_str(), "wb");
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::CreateImage(void)
{
    LOG_DBG(m_name + ": CreateImage\n");
    /****
    NvSciRmGpuId gpuId;
    CUuuid uuid;
    int m_cudaDeviceId = 0;
    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");
    cudaStatus = cudaSetDevice(m_cudaDeviceId);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");
    auto cudaErr = cuDeviceGetUuid(&uuid, m_cudaDeviceId);
    CHK_CUDAERR_AND_RETURN(cudaErr, "cuDeviceGetUuid");
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));
    ***/
    NvSciError sciErr;
    NvMediaStatus mediaErr;
    NvSciBufAttrList bufAttr = NULL;
    sciErr = NvSciBufAttrListCreate(m_sciBufModule, &bufAttr);
    CHK_NVSCISTATUS_AND_RETURN(sciErr," image NvSciBufAttrListCreate");
    NvSciBufType bufType = NvSciBufType_Image;
    uint32_t planeCount = 1;
    NvSciBufAttrValColorFmt planeColorFormat = NvSciColor_A8B8G8R8;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    NvSciBufAttrValColorStd colorStd = NvSciColorStd_SRGB;
    bool enableCpuCache = true;
    bool needCpuAccess = true;
    uint64_t topPadding = 0;
    uint64_t bottomPadding = 0;
    uint32_t width = (uint32_t)m_Width;
    uint32_t height =  (uint32_t)m_Height;
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrKeyValuePair bufAllocKeyVal[] = {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufImageAttrKey_PlaneColorFormat, &planeColorFormat, sizeof(NvSciBufAttrValColorFmt)*planeCount},
            //{ NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) },
            { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
            { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
            { NvSciBufImageAttrKey_PlaneColorStd,&colorStd,sizeof(NvSciBufAttrValColorStd) * planeCount},
            { NvSciBufImageAttrKey_PlaneWidth, &width, sizeof(width) },
            { NvSciBufImageAttrKey_PlaneHeight, &height, sizeof(height) },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess) },
            { NvSciBufGeneralAttrKey_EnableCpuCache, &enableCpuCache, sizeof(enableCpuCache) },
            { NvSciBufImageAttrKey_TopPadding, &topPadding, sizeof(topPadding) },
            { NvSciBufImageAttrKey_BottomPadding, &bottomPadding, sizeof(bottomPadding) },
            { NvSciBufImageAttrKey_ScanType, &scanType, sizeof(scanType) }
    };
    sciErr = NvSciBufAttrListSetAttrs(bufAttr, bufAllocKeyVal, sizeof(bufAllocKeyVal) / sizeof(NvSciBufAttrKeyValuePair));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "image NvSciBufAttrListSetAttrs");

    

    mediaErr = NvMedia2DFillNvSciBufAttrList(m_pNvMedia, bufAttr);
    PCHK_NVMSTATUS_AND_RETURN(mediaErr, "NvMedia2DFillNvSciBufAttrList");

    NvSciBufAttrList conflicts = NULL,reconciledAttrList = NULL;
    sciErr = NvSciBufAttrListReconcile(&bufAttr, 1,
                                       &reconciledAttrList, &conflicts);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "image NvSciBufAttrListReconcile");
    sciErr = NvSciBufObjAlloc(reconciledAttrList, &outputImage);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "image NvSciBufObjAlloc");

    if (NULL != conflicts) {
        NvSciBufAttrListFree(conflicts);
    }

    if (NULL != reconciledAttrList) {
        NvSciBufAttrListFree(reconciledAttrList);
    }
    
    if (NULL != bufAttr) {
        NvSciBufAttrListFree(bufAttr);
    }

    mediaErr =  NvMedia2DRegisterNvSciBufObj(m_pNvMedia,outputImage);
    PCHK_NVMSTATUS_AND_RETURN(mediaErr, "NvMedia2DRegisterNvSciBufObj");
    // auto status = SetEncodeConfig();
    return NVSIPL_STATUS_OK;
}

CNvMediaConsumer::~CNvMediaConsumer(void)
{
    LOG_DBG("CNvMediaConsumer release.\n");

    for (NvSciBufObj bufObj : m_pSciBufObjs) {
        if (bufObj != nullptr) {
            NvMedia2DUnregisterNvSciBufObj(m_pNvMedia, bufObj);
        }
    }
    for (NvSciBufObj bufObj : m_pSciBufObjs) {
        if (bufObj != nullptr) {
            NvSciBufObjFree(bufObj);
        }
    }
    UnregisterSyncObjs();
    
    if (outputImage != nullptr) {
        NvMedia2DUnregisterNvSciBufObj(m_pNvMedia, outputImage);
        NvSciBufObjFree(outputImage);
    }
    if (m_pNvMedia != nullptr) {
        NvMedia2DDestroy(m_pNvMedia);
    }
    if(m_pOutputTimeFile != nullptr){
        fflush(m_pOutputTimeFile);
        fclose(m_pOutputTimeFile);
    }
    if(m_pOutputImgFile != nullptr){
        fflush(m_pOutputImgFile);
        fclose(m_pOutputImgFile);
    }
}

// Buffer setup functions
SIPLStatus CNvMediaConsumer::SetDataBufAttrList(NvSciBufAttrList &bufAttrList) {
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    status = NvMedia2DFillNvSciBufAttrList(m_pNvMedia, bufAttrList);
    PCHK_NVMSTATUS_AND_RETURN(status, "NvMedia2DFillNvSciBufAttrList");
    
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    bool cpuaccess_flag = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
    };

    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

// Sync object setup functions
SIPLStatus CNvMediaConsumer::SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList)
{
    auto nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_pNvMedia, signalerAttrList, NVMEDIA_SIGNALER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Signaler NvMedia2DFillNvSciSyncAttrList");

    nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_pNvMedia, waiterAttrList, NVMEDIA_WAITER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Waiter NvMedia2DFillNvSciSyncAttrList");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    if (m_pNvMedia) {
        PLOG_DBG("Mapping data buffer, packetIndex: %u.\n", packetIndex);
        auto sciErr = NvSciBufObjDup(bufObj, &m_pSciBufObjs[packetIndex]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjDup");

        NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_pNvMedia, m_pSciBufObjs[packetIndex]);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::RegisterSignalSyncObj(void)
{
    auto nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_pNvMedia, NVMEDIA_EOFSYNCOBJ, m_signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciSyncObj for EOF");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::RegisterWaiterSyncObj(uint32_t index)
{
    auto nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_pNvMedia, NVMEDIA_PRESYNCOBJ, m_waiterSyncObjs[index]);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciSyncObj for PRE");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::UnregisterSyncObjs(void)
{
    auto nvmStatus = NvMedia2DUnregisterNvSciSyncObj(m_pNvMedia,  m_signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPUnregisterNvSciSyncObj for EOF");

    for (uint32_t i = 0U; i  < m_numWaitSyncObj; i++) {
        auto nvmStatus = NvMedia2DUnregisterNvSciSyncObj(m_pNvMedia, m_waiterSyncObjs[i]);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPUnregisterNvSciSyncObj for PRE");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{   
    auto mediaErr = NvMedia2DGetComposeParameters(m_pNvMedia,
                                               &media2DParam);
    PCHK_NVMSTATUS_AND_RETURN(mediaErr, "NvMedia2DGetComposeParameters");
    auto nvmStatus = NvMedia2DInsertPreNvSciSyncFence(m_pNvMedia,media2DParam, &prefence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DInsertPreNvSciSyncFence");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::SetEofSyncObj(void)
{
    auto nvmStatus = NvMedia2DSetNvSciSyncObjforEOF(m_pNvMedia,media2DParam, m_signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetNvSciSyncObjforEOF");

    return NVSIPL_STATUS_OK;
}


SIPLStatus CNvMediaConsumer::ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) {    
    PLOG_DBG("Process payload (packetIndex = 0x%x).\n", packetIndex);
    auto nvmStatus = NvMedia2DSetSrcNvSciBufObj(m_pNvMedia, media2DParam, 0, m_pSciBufObjs[packetIndex]);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetSrcNvSciBufObj");

    nvmStatus = NvMedia2DSetDstNvSciBufObj(m_pNvMedia, media2DParam, outputImage);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetDstNvSciBufObj");

    NvMedia2DComposeResult result;
    nvmStatus = NvMedia2DCompose(m_pNvMedia, media2DParam, &result);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCompose");
    
    nvmStatus = NvMedia2DGetEOFNvSciSyncFence(m_pNvMedia, &result, pPostfence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPGetEOFNvSciSyncFence");
    PLOG_DBG("ProcessPayload succ.\n");
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMediaConsumer::OnProcessPayloadDone(uint32_t packetIndex) {
    //deal outputImage
    if (m_consConfig.bFileDump && (m_frameNum >= DUMP_START_FRAME && m_frameNum < DUMP_START_FRAME+20)) {
        void* va_ptr = nullptr;
        auto sciErr = NvSciBufObjGetConstCpuPtr(outputImage, (const void**)&va_ptr);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetConstCpuPtr");
        #if 1
        uint8_t* basePtr = static_cast<uint8_t*>(va_ptr);
        uint32_t imageSize = m_Width*m_Height*4;//
        auto ret = fwrite(basePtr, 1, imageSize, m_pOutputImgFile);
        fflush(m_pOutputImgFile);
        if(ret != imageSize){
            PLOG_ERR("Error writing image cpu file\n");
            return NVSIPL_STATUS_ERROR;
        }
        #endif
        uint64_t encodedTimeStamp = GetCurrentPTPTimeMicroSec();
        uint64_t captureTime = 0;
        if(m_metaPtrs[packetIndex] != nullptr){
            captureTime = m_metaPtrs[packetIndex]->captureImgTimestamp;
        } 
        if(fprintf(m_pOutputTimeFile,"%d_%ld_%ld_%ld\n",m_metaPtrs[packetIndex]->streamImageCount,captureTime,encodedTimeStamp,encodedTimeStamp-captureTime) < 0){
            PLOG_ERR("Error writing time file\n");
            return NVSIPL_STATUS_ERROR;
        }
    }

    if(m_frameNum == DUMP_START_FRAME+20){
        LOG_MSG("Dump sensor_%d image file finish\n",m_uSensorId);
    }
    
    return NVSIPL_STATUS_OK;
}

bool CNvMediaConsumer::ToSkipFrame(uint32_t frameNum){
    if(frameNum < DUMP_START_FRAME){
        return true;
    }
    return false;
}

}
}
}
