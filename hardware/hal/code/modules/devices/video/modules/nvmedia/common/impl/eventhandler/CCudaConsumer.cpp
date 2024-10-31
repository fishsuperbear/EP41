// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CCudaConsumer.hpp"
#include "hw_nvmedia_eventhandler_impl.h"
#include "gpu_convert.hpp"
#include <cassert>
#include "netacuda/netacuda.h"

#define CUDACONSUMER_STRIDE ALIGN_STEP

CCudaConsumer::CCudaConsumer(NvSciStreamBlock handle, u32 uSensor, u32 i_blockindex, u32 i_sensorindex, NvSciStreamBlock queueHandle, u32 i_capturewidth, u32 i_captureheight)
    : CConsumer("CudaConsumer", handle, uSensor, queueHandle, CUDA_CONSUMER) {
    m_streamWaiter = nullptr;
    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++)
    {
        m_extMem[i] = 0U;
        m_pCudaCopyMem[i] = nullptr;
        m_devPtr[i] = nullptr;
    }

    m_signalerSem = 0U;
    m_waiterSem = 0U;
    m_hostBufLen = 0;
    m_pHostBuf = nullptr;
    m_FirstCall = true;

    _capturewidth = i_capturewidth;
    _captureheight = i_captureheight;
    _blockindex = i_blockindex;
    _sensorindex = i_sensorindex;
}

SIPLStatus CCudaConsumer::GetBufAttrList(NvSciBufAttrList bufAttrList)
{
    neta_cuda_init(1);

    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

    NvSciRmGpuId gpuId;
    CUuuid uuid;
    auto cudaErr = cuDeviceGetUuid(&uuid, 0);
    CHK_CUDAERR_AND_RETURN(cudaErr, "cuDeviceGetUuid");
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    bool cpuaccess_flag = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
    };

    NvSciError sciErr = NvSciBufAttrListSetAttrs(bufAttrList, bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    return NVSIPL_STATUS_OK;
}

hw_ret_s32 CCudaConsumer::RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t *i_pcbconfig,
                                           HWNvmediaEventHandlerRegDataCbConfig *i_peventhandlercbconfig)
{
    _datacb = i_pcbconfig->cudacb;
    /*
     * Currently only support sync mode.
     */
    if (!i_pcbconfig->bsynccb)
    {
        HW_NVMEDIA_LOG_ERR("Only support data cb SYNC mode!\r\n");
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_SYNCCB_MODE);
    }
    _pcontext = i_pcbconfig->pcustom;

    switch (i_pcbconfig->type)
    {
    case HW_VIDEO_REGDATACB_TYPE_CUDA:
        switch (i_pcbconfig->cudaconfig.imgtype)
        {
        case HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NHWC_BGR:
            break;
        case HW_VIDEO_REGCUDADATACB_IMGTYPE_YUYV:
            // currently not support, may support it later
            return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_UNEXPECTED);

        default:
            return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_UNEXPECTED);
        }
        break;

    default:
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_UNEXPECTED);
    }

    _cudaconfig = i_pcbconfig->cudaconfig;
    switch (_cudaconfig.imgtype)
    {
    case HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NHWC_BGR:
        _imgtype = i_pcbconfig->cudaconfig.imgtype;
        _interpolation = i_pcbconfig->cudaconfig.interpolation;
        if (i_pcbconfig->busecaptureresolution)
        {
            _width = _capturewidth;
            _height = _captureheight;
        }
        else
        {
            _width = i_pcbconfig->customwidth;
            _height = i_pcbconfig->customheight;
        }
        _busecaptureframerate = i_pcbconfig->busecaptureframerate;
        _customframerate = i_pcbconfig->customframerate;
        _gpuimage.out_img_type = GPU_IMG_TYPE::GPU_Bayer_RGB888;
        _gpuimage.image = (void *)gpuutils::create_rgb_gpu_image(_width, _height, 1, PixelLayout::NHWC_BGR, netaos::gpu::DataType::Uint8);
        // for ~CCudaConsumer release only
        _pimage = ((RGBGPUImage *)_gpuimage.image)->data;
        // auto bytes = width * height * channel * batch * dtype_sizeof(dtype);
        _buffsize = _width * _height * 3 * 1 * 1;
        _pgpuimage = &_gpuimage;
        break;
    default:
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_UNEXPECTED);
    }
    return 0;
}

SIPLStatus CCudaConsumer::InitCuda(void)
{
    // Init CUDA
    neta_cuda_init(1);

    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");
    
    cudaStatus = cudaStreamCreateWithFlags(&m_streamWaiter, cudaStreamNonBlocking);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaStreamCreateWithFlags");
    PLOG_DBG("%s:Created consumer's CUDA stream.\r\n", m_name.c_str());

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::HandleClientInit(void)
{
    auto status = InitCuda();
    PCHK_STATUS_AND_RETURN(status, "InitCuda");

    if (m_consConfig.bFileDump)
    {
        string fileName = "multicast_cuda" + to_string(m_uSensorId) + ".yuv";
        m_pOutputFile = fopen(fileName.c_str(), "wb");
        PCHK_PTR_AND_RETURN(m_pOutputFile, "Open CUDA output file");
    }

    m_numWaitSyncObj = 1U;

    return NVSIPL_STATUS_OK;
}

CCudaConsumer::~CCudaConsumer(void)
{
    PLOG_DBG("%s:release.\r\n", m_name.c_str());

    if (_plPtr)
    {
        cudaFree(_plPtr);
    }
    if (_pimage != nullptr)
    {
        cudaFree(_pimage);
    }

    if (m_pOutputFile != nullptr)
    {
        fflush(m_pOutputFile);
        fclose(m_pOutputFile);
    }
    if (m_waiterSem != nullptr)
    {
        (void)cudaDestroyExternalSemaphore(m_waiterSem);
        m_waiterSem = nullptr;
    }
    if (m_signalerSem != nullptr)
    {
        cudaDestroyExternalSemaphore(m_signalerSem);
        m_signalerSem = nullptr;
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++)
    {
        cudaFree(m_devPtr[i]);
        cudaDestroyExternalMemory(m_extMem[i]);
        if (m_pCudaCopyMem[i])
        {
            cudaFreeHost(m_pCudaCopyMem[i]);
            m_pCudaCopyMem[i] = nullptr;
        }
    }
#if (HW_NVMEDIA_MAY_CHANGE_ME_LATER == 1)
    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++)
    {
        for (uint32_t j = 0U; j < MAX_NUM_SURFACES; j++)
        {
            if (m_mipmapArray[i][j] != nullptr)
            {
                cudaFreeMipmappedArray(m_mipmapArray[i][j]);
            }
        }
    }
#endif

    cudaStreamDestroy(m_streamWaiter);
}

SIPLStatus CCudaConsumer::SetDataBufAttrList(NvSciBufAttrList &bufAttrList)
{
    NvSciRmGpuId gpuId;
    CUuuid uuid;
    auto cudaErr = cuDeviceGetUuid(&uuid, m_cudaDeviceId);
    CHK_CUDAERR_AND_RETURN(cudaErr, "cuDeviceGetUuid");
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));

    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    bool cpuaccess_flag = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {{NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId)},
                                           {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
                                           {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
                                           {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
                                            sizeof(cpuaccess_flag)}};

    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    PLOG_DBG("%s:Set buf attribute list succeed.\r\n", m_name.c_str());

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::GetSyncWaiterAttrList(NvSciSyncAttrList waiterAttrList)
{
    assert(waiterAttrList != nullptr);

    auto cudaStatus = cudaDeviceGetNvSciSyncAttributes(waiterAttrList, 0, cudaNvSciSyncAttrWait);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList)
{
    auto cudaStatus = cudaDeviceGetNvSciSyncAttributes(signalerAttrList, m_cudaDeviceId, cudaNvSciSyncAttrSignal);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");
    PLOG_DBG("%s:Set CUDA-signaler attribute value.\r\n", m_name.c_str());

    cudaStatus = cudaDeviceGetNvSciSyncAttributes(waiterAttrList, m_cudaDeviceId, cudaNvSciSyncAttrWait);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");
    PLOG_DBG("%s:Set CUDA-waiter attribute value.\r\n", m_name.c_str());

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    auto status = PopulateBufAttr(bufObj, m_bufAttrs[packetIndex]);
    PCHK_STATUS_AND_RETURN(status, "PopulateBufAttr");

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = bufObj;
    memHandleDesc.size = m_bufAttrs[packetIndex].size;
    auto cudaStatus = cudaImportExternalMemory(&m_extMem[packetIndex], &memHandleDesc);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalMemory");

    // Only BL is supported, since it is the default memory layout
    if (m_bufAttrs[packetIndex].layout == NvSciBufImage_BlockLinearType)
    {
        PLOG_DBG("%s:MapDataBuffer, layout is NV12BlockLinear.\r\n", m_name.c_str());
        struct cudaExtent extent[MAX_NUM_SURFACES];
        struct cudaChannelFormatDesc desc[MAX_NUM_SURFACES];
        struct cudaExternalMemoryMipmappedArrayDesc mipmapDesc[MAX_NUM_SURFACES];
        (void *)memset(extent, 0, MAX_NUM_SURFACES * sizeof(struct cudaExtent));
        (void *)memset(desc, 0, MAX_NUM_SURFACES * sizeof(struct cudaChannelFormatDesc));
        (void *)memset(mipmapDesc, 0, MAX_NUM_SURFACES * sizeof(struct cudaExternalMemoryMipmappedArrayDesc));
        for (uint32_t planeId = 0; planeId < m_bufAttrs[packetIndex].planeCount; planeId++)
        {
            /* Setting for each plane buffer
             * SP format has 2 planes
             * Planar format has 3 planes  */
            /* extent[planeId].width = m_bufAttrs[packetIndex].planeWidths[planeId]; */
            /* extent[planeId].height = m_bufAttrs[packetIndex].planeHeights[planeId]; */
            /* extent[planeId].depth = 1; */
            // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaExtent.html#structcudaExtent
            // Width in elements when referring to array memory, in bytes when referring to linear memory
            // NvSciBufImageAttrKey_PlanePitch: Outputs the pitch (aka width in bytes) for every plane.
            // Bug 3880762
            extent[planeId].width = m_bufAttrs[packetIndex].planePitches[planeId] / (m_bufAttrs[packetIndex].planeBitsPerPixels[planeId] / 8);
            extent[planeId].height = m_bufAttrs[packetIndex].planeAlignedHeights[planeId];
            // Set the depth to 0 will create a 2D mapped array,
            // set the depth to 1 which indicates CUDA that it is a 3D array
            // Bug 3907432
            extent[planeId].depth = 0;

            /* For Y */
            if (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_Y8)
            {
                desc[planeId] = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
            }
            /* For UV */
            if ((m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_U8V8) ||
                (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_U8_V8) ||
                (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_V8U8) ||
                (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_V8_U8))
            {
                desc[planeId] = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
            }
            mipmapDesc[planeId].offset = m_bufAttrs[packetIndex].planeOffsets[planeId];
            mipmapDesc[planeId].formatDesc = desc[planeId];
            mipmapDesc[planeId].extent = extent[planeId];
            mipmapDesc[planeId].flags = 0;
            mipmapDesc[planeId].numLevels = 1;
            cudaStatus = cudaExternalMemoryGetMappedMipmappedArray(&m_mipmapArray[packetIndex][planeId], m_extMem[packetIndex], &mipmapDesc[planeId]);
            CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaExternalMemoryGetMappedMipmappedArray");
        } /* end for */
    }
    else if (m_bufAttrs[packetIndex].layout == NvSciBufImage_PitchLinearType)
    {
        PLOG_INFO("%s:MapDataBuffer, layout is NV12PitchLinear.\r\n", m_name.c_str());
        struct cudaExternalMemoryBufferDesc memBufferDesc = {0, m_bufAttrs[packetIndex].size, 0};
        auto cudaStatus = cudaExternalMemoryGetMappedBuffer(&m_devPtr[packetIndex], m_extMem[packetIndex], &memBufferDesc);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaExternalMemoryGetMappedBuffer");
    }
    else
    {
        PLOG_ERR("%s:Unsupported layout\r\n", m_name.c_str());
        return NVSIPL_STATUS_ERROR;
    }
    if (m_pCudaCopyMem[packetIndex] == nullptr)
    {
        cudaStatus = cudaHostAlloc((void **)&m_pCudaCopyMem[packetIndex], m_bufAttrs[packetIndex].size, cudaHostAllocDefault);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaHostAlloc");
        PCHK_PTR_AND_RETURN(m_pCudaCopyMem[packetIndex], "m_pCudaCopyMem allocation");
        (void *)memset(m_pCudaCopyMem[packetIndex], 0, m_bufAttrs[packetIndex].size);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::RegisterSignalSyncObj(NvSciSyncObj signalSyncObj)
{
    // Map CUDA Signaler sync objects
    cudaExternalSemaphoreHandleDesc extSemDescSig;
    memset(&extSemDescSig, 0, sizeof(extSemDescSig));
    extSemDescSig.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDescSig.handle.nvSciSyncObj = signalSyncObj;
    auto cudaStatus = cudaImportExternalSemaphore(&m_signalerSem, &extSemDescSig);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalSemaphore");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj)
{
    neta_cuda_init(1);
    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = waiterSyncObj;
    auto cudaStatus = cudaImportExternalSemaphore(&m_waiterSem, &extSemDesc);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalSemaphore");

    return NVSIPL_STATUS_OK;
}

// Before calling PreSync, m_nvmBuffers[packetIndex] should already be filled.
SIPLStatus CCudaConsumer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{
    /* Instruct CUDA to wait for the producer fence */
    if (m_FirstCall)
    {
        neta_cuda_init(1);
        size_t unused;
        auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");
        m_FirstCall = false;
    }
    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    waitParams.params.nvSciSync.fence = &prefence;
    waitParams.flags = 0;
    auto cudaStatus = cudaWaitExternalSemaphoresAsync(&m_waiterSem, &waitParams, 1, m_streamWaiter);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cuWaitExternalSemaphoresAsync");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::BlToPlConvert(uint32_t packetIndex, void *dstptr)
{
    uint8_t *vaddr_plane[2] = {nullptr};

    auto cudaStatus = cudaGetMipmappedArrayLevel(&m_mipLevelArray[packetIndex][0U], m_mipmapArray[packetIndex][0], 0U);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");
    cudaStatus = cudaGetMipmappedArrayLevel(&m_mipLevelArray[packetIndex][1U], m_mipmapArray[packetIndex][1], 0U);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");

    vaddr_plane[0] = (uint8_t *)&m_mipLevelArray[packetIndex][0U];
    vaddr_plane[1] = (uint8_t *)&m_mipLevelArray[packetIndex][1U];
    cudaStatus = cudaMemcpy2DFromArrayAsync(dstptr,
                                            (size_t)m_bufAttrs[packetIndex].planeWidths[0U],
                                            *((cudaArray_const_t *)vaddr_plane[0]),
                                            0, 0,
                                            (size_t)m_bufAttrs[packetIndex].planeWidths[0U],
                                            (size_t)m_bufAttrs[packetIndex].planeHeights[0U],
                                            cudaMemcpyDeviceToDevice,
                                            m_streamWaiter);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2DFromArrayAsync for plane 0");
    /* cudaStreamSynchronize(m_streamWaiter); */

    uint8_t *second_dst = (uint8_t *)dstptr +
                          (size_t)(m_bufAttrs[packetIndex].planeWidths[0U] * m_bufAttrs[packetIndex].planeHeights[0U]);
    cudaStatus = cudaMemcpy2DFromArrayAsync((void *)(second_dst),
                                            (size_t)m_bufAttrs[packetIndex].planeWidths[0U],
                                            *((cudaArray_const_t *)vaddr_plane[1]),
                                            0, 0,
                                            (size_t)m_bufAttrs[packetIndex].planeWidths[0U],
                                            (size_t)(m_bufAttrs[packetIndex].planeHeights[0U] / 2),
                                            cudaMemcpyDeviceToDevice,
                                            m_streamWaiter);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2DFromArrayAsync for plane 1");
    /* cudaStreamSynchronize(m_streamWaiter); */

    return NVSIPL_STATUS_OK;
}
SIPLStatus CCudaConsumer::ProcessPayload(uint32_t packetIndex,
                                         NvSciSyncFence *pPostfence)
{
    // printf("CCudaConsumer::ProcessPayload+++++++++++++++\r\n"); 
    /* printf("get buffer format=%d",m_bufAttrs[packetIndex].planeColorFormats[0]); */
    // printf("CCudaConsumer::ProcessPayload get buffer .layout=%d",m_bufAttrs[packetIndex].layout);
    // printf("CCudaConsumer::ProcessPayload get buffer .NvSciBufImage_BlockLinearType=%d;",NvSciBufImage_BlockLinearType);
    SIPLStatus status = NVSIPL_STATUS_OK;
    if (_pgpuimage == nullptr)
    {
        return status;
    }
    int inputWidth = m_bufAttrs[packetIndex].planeWidths[0];
    int inputHeight = m_bufAttrs[packetIndex].planeHeights[0];
    YUVGPUImage *input = nullptr;
    FillColor color;
    // memset(&color, 0, sizeof(color));
    color.color[0] = 0;
    color.color[1] = 255;
    color.color[2] = 128;
    if (m_bufAttrs[packetIndex].layout == NvSciBufImage_BlockLinearType)
    {
        /* printf("CCudaConsumer::NvSciBufImage_BlockLinearType+++++++++++++++\r\n"); */
        // todo
        size_t uNumBytes = (size_t)(m_bufAttrs[packetIndex].planeWidths[0] * (int)m_bufAttrs[packetIndex].planeHeights[0] * 1.5);
        if (_plPtr == nullptr)
        {
            auto cudaStatus = cudaMalloc((void **)&_plPtr, uNumBytes);
            if (cudaStatus != cudaSuccess)
            {
                PLOG_ERR("%s:cudaMalloc failed: %u\r\n", m_name.c_str(), status);
                return NVSIPL_STATUS_OUT_OF_MEMORY;
            }
        }
        status = BlToPlConvert(packetIndex, (void *)_plPtr);
        input = gpuutils::create_yuv_gpu_image(inputWidth, inputHeight, 1,
                                               YUVFormat::NV12PitchLinear, (void *)_plPtr, (u8 *)_plPtr + inputWidth * inputHeight);
    }
    else
    { // pl
        if (m_bufAttrs[packetIndex].planeColorFormats[0] == NvSciColor_Y8U8Y8V8)
        { // NvSciColor_Y8U8Y8V8 //yuyv 021

            input = gpuutils::create_yuv_gpu_image(inputWidth, inputHeight, 1,
                                                   YUVFormat::YUV422Packed_YUYV_PitchLinear, (void *)m_devPtr[packetIndex]);
        }
        else
        { // NV12 PL
            input = gpuutils::create_yuv_gpu_image(inputWidth, inputHeight, 1,
                                                   YUVFormat::NV12PitchLinear, (void *)m_devPtr[packetIndex], (u8 *)m_devPtr[packetIndex] + m_bufAttrs[packetIndex].planeOffsets[1]);
        }
    }
    RGBGPUImage *output = nullptr;
    switch (_pgpuimage->out_img_type)
    {
    default:
    case GPU_IMG_TYPE::GPU_Bayer_RGB888:
        // m_bufAttrs[packetIndex].planeColorFormats[0];
        output = (RGBGPUImage *)_pgpuimage->image;
        // Bilinear cost time but effect is good, Nearest cost time less but effect is not so good
        gpuutils::batched_convert_yuv_to_rgb(input, output, output->width, output->height, 0, 0, color, 0, 0, 0, 1, 1, 1, Interpolation::Nearest, m_streamWaiter);
        // gpuutils::save_rgbgpu_to_file("test.rgb", output, m_streamWaiter);
        cudaStreamSynchronize(m_streamWaiter);
        break;
    case GPU_IMG_TYPE::GPU_YUV422_YUYV:
        break;
    case GPU_IMG_TYPE::GPU_YUV420_NV12:
        break;
    }
    struct hw_video_cudabufferinfo_t datacbinfo;
    // Convert NVIDIA internal timestamp to milliseconds
    // nv_timestamp: NVIDIA internal timestamp value
    // ms: Converted timestamp in milliseconds
    // Formula: nv_timestamp * 32 / 1000000 = ms
    // Here, we are using nanoseconds as the unit

    // Code Example:
    // uint64_t nv_timestamp = m_metaPtrs[packetIndex]->frameCaptureTSC; // NVIDIA internal timestamp
    // uint64_t ms = (nv_timestamp * 32) / 1 000 000;
    datacbinfo.timeinfo.framecapturetsc = static_cast<MetaData *>(m_metaPtrs[packetIndex])->frameCaptureTSC * 32;
    datacbinfo.timeinfo.framecapturestarttsc = static_cast<MetaData *>(m_metaPtrs[packetIndex])->frameCaptureStartTSC * 32;
    datacbinfo.timeinfo.exposurestarttime = (static_cast<MetaData *>(m_metaPtrs[packetIndex])->frameCaptureStartTSC * 32)
        - (uint64_t)((static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[0]) * 1000000000)
        - (uint64_t)((static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[2]) * 1000000000);

#if 0
    printf("CCudaConsumer m_uSensorId:%d frameCaptureTSC:%ld frameCaptureStartTSC:%ld\n", m_uSensorId,
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->frameCaptureTSC,
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->frameCaptureStartTSC);
    printf("CCudaConsumer expTimeValid:%d exposureTime[%f][%f][%f][%f]\n",
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[0],
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[1],
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[2],
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[3]);

    printf("CCudaConsumer exposureTime[0]:%f exposureTime[2]:%f exposureTime[0]:%ld exposureTime[2]:%ld\n",
        (static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[0]) * 1000000000,
        (static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[2]) * 1000000000,
        (uint64_t)((static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[0]) * 1000000000),
        (uint64_t)((static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime[2]) * 1000000000));

    printf("CCudaConsumer framecapturetsc:%ld framecapturestarttsc:%ld exposurestarttime:%ld\n", 
        datacbinfo.timeinfo.framecapturetsc, datacbinfo.timeinfo.framecapturestarttsc,
        datacbinfo.timeinfo.exposurestarttime);
#endif

    datacbinfo.blockindex = _blockindex;
    datacbinfo.sensorindex = _sensorindex;
    datacbinfo.outputtype = (HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE)_outputtype;
    datacbinfo.imgtype = _imgtype;
    datacbinfo.interpolation = _interpolation;
    datacbinfo.capturewidth = _capturewidth;
    datacbinfo.captureheight = _captureheight;
    datacbinfo.width = _width;
    datacbinfo.height = _height;
    datacbinfo.stride = CUDACONSUMER_STRIDE;
    datacbinfo.rotatedegrees = 0;
    datacbinfo.gpuinfo.imgtype = _imgtype;
    switch (_imgtype)
    {
    case HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NHWC_BGR:
        datacbinfo.gpuinfo.rgbinfo.imgtype = _imgtype;
        datacbinfo.gpuinfo.rgbinfo.pbuff = ((RGBGPUImage *)_pgpuimage->image)->data;
        datacbinfo.gpuinfo.rgbinfo.buffsize = _buffsize;
        break;
    default:
        break;
    }
    datacbinfo.bsynccb = 1;
    datacbinfo.bneedfree = 0;
    datacbinfo.pcustom = _pcontext;
    /*
     * Call the user data callback here.
     */
    _datacb(&datacbinfo);

    return status;
}

SIPLStatus CCudaConsumer::OnProcessPayloadDone(uint32_t packetIndex)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    // dump frames to local file
    if (m_consConfig.bFileDump && (m_frameNum >= DUMP_START_FRAME && m_frameNum <= DUMP_END_FRAME))
    {
        if (nullptr != m_pHostBuf && m_hostBufLen > 0)
        {
            char *buffer = (char *)m_pHostBuf;
            // auto ret = fwrite(buffer, 1, m_hostBufLen, m_pOutputFile);
            for (uint32_t j = 0U; j < m_bufAttrs[packetIndex].planeHeights[0]; j++)
            {
                if (j * m_bufAttrs[packetIndex].planePitches[0] < m_hostBufLen)
                    fwrite(((uint8_t *)buffer) + j * m_bufAttrs[packetIndex].planePitches[0],
                           m_bufAttrs[packetIndex].planeWidths[0] * m_bufAttrs[packetIndex].planeBitsPerPixels[0] / 8,
                           1U,
                           m_pOutputFile);
            }
        }
    }

    // cleanup:
    if (nullptr != m_pHostBuf)
    {
        free(m_pHostBuf);
        m_pHostBuf = nullptr;
    }
    m_hostBufLen = 0;

    return status;
}

SIPLStatus CCudaConsumer::UnregisterSyncObjs(void)
{
    cudaDestroyExternalSemaphore(m_waiterSem);
    cudaDestroyExternalSemaphore(m_signalerSem);

    return NVSIPL_STATUS_OK;
}
