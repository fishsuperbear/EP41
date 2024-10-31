// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CCudaConsumer.hpp"

CCudaConsumer::CCudaConsumer(NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle):
    CConsumer("CudaConsumer", handle, uSensor, queueHandle)
{
    m_streamWaiter = nullptr;
    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++) {
        m_extMem[i] = 0U;
        m_pCudaCopyMem[i] = nullptr;
        m_devPtr[i] = nullptr;
    }

    m_signalerSem = 0U;
    m_waiterSem = 0U;
    m_hostBufLen = 0;
    m_pHostBuf = nullptr;
    m_FirstCall = true;
}

SIPLStatus CCudaConsumer::InitCuda(void)
{
    // Init CUDA
    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

    cudaStatus = cudaSetDevice(m_cudaDeviceId);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");

    cudaStatus = cudaStreamCreateWithFlags(&m_streamWaiter, cudaStreamNonBlocking);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaStreamCreateWithFlags");
    PLOG_DBG("Created consumer's CUDA stream.\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::HandleClientInit(void)
{
    auto status = InitCuda();
    PCHK_STATUS_AND_RETURN(status, "InitCuda");

    if (m_consConfig.bFileDump) {
        string fileName = "multicast_cuda" + to_string(m_uSensorId) + ".yuv";
        m_pOutputFile = fopen(fileName.c_str(), "wb");
        PCHK_PTR_AND_RETURN(m_pOutputFile, "Open CUDA output file");
    }

    m_numWaitSyncObj = 1U;

    return NVSIPL_STATUS_OK;
}

CCudaConsumer::~CCudaConsumer(void)
{
    PLOG_DBG("release.\n");

    if (m_pOutputFile != nullptr) {
        fflush(m_pOutputFile);
        fclose(m_pOutputFile);
    }
    if (m_waiterSem != nullptr) {
        (void)cudaDestroyExternalSemaphore(m_waiterSem);
        m_waiterSem = nullptr;
    }
    if (m_signalerSem != nullptr) {
        cudaDestroyExternalSemaphore(m_signalerSem);
        m_signalerSem = nullptr;
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++) {
        cudaFree(m_devPtr[i]);
        cudaDestroyExternalMemory(m_extMem[i]);
        if (m_pCudaCopyMem[i]) {
            cudaFreeHost(m_pCudaCopyMem[i]);
            m_pCudaCopyMem[i] = nullptr;
        }
    }
    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++) {
        for (uint32_t j = 0U; j < m_bufAttrs[i].planeCount; j++) {
            cudaFreeMipmappedArray(m_mipmapArray[i][j]);
        }
    }

    cudaStreamDestroy(m_streamWaiter);
}

SIPLStatus CCudaConsumer::SetDataBufAttrList(void)
{
    NvSciRmGpuId gpuId;
    CUuuid uuid;
    auto cudaErr = cuDeviceGetUuid(&uuid, m_cudaDeviceId);
    CHK_CUDAERR_AND_RETURN(cudaErr, "cuDeviceGetUuid");
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));

    NvSciBufAttrKeyValuePair genbuffattrs[] = {
        { NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) },
    };
    auto sciErr = NvSciBufAttrListSetAttrs(m_bufAttrLists[DATA_ELEMENT_INDEX], genbuffattrs, sizeof(genbuffattrs) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    bool cpuaccess_flag = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
    };

    sciErr = NvSciBufAttrListSetAttrs(m_bufAttrLists[DATA_ELEMENT_INDEX], bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    PLOG_DBG("Set buf attribute list succeed.\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::SetSyncAttrList(void)
{
    auto cudaStatus = cudaDeviceGetNvSciSyncAttributes(m_signalerAttrList,
                                                     m_cudaDeviceId,
                                                     cudaNvSciSyncAttrSignal);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");
    PLOG_DBG("Set CUDA-signaler attribute value.\n");

    cudaStatus = cudaDeviceGetNvSciSyncAttributes(m_waiterAttrList,
                                                  m_cudaDeviceId,
                                                  cudaNvSciSyncAttrWait);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");
    PLOG_DBG("Set CUDA-waiter attribute value.\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::MapDataBuffer(uint32_t packetIndex)
{
    auto status = PopulateBufAttr(m_packets[packetIndex].dataObj, m_bufAttrs[packetIndex]);
    PCHK_STATUS_AND_RETURN(status, "PopulateBufAttr");

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = m_packets[packetIndex].dataObj;
    memHandleDesc.size = m_bufAttrs[packetIndex].size;
    auto cudaStatus = cudaImportExternalMemory(&m_extMem[packetIndex], &memHandleDesc);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalMemory");

    //Only BL is supported, since it is the default memory layout
    if(m_bufAttrs[packetIndex].layout == NvSciBufImage_BlockLinearType) {
        PLOG_DBG("MapDataBuffer, layout is blockLinear.\n");
        struct cudaExtent extent[MAX_NUM_SURFACES];
        struct cudaChannelFormatDesc desc[MAX_NUM_SURFACES];
        struct cudaExternalMemoryMipmappedArrayDesc mipmapDesc[MAX_NUM_SURFACES];
        (void*)memset(extent, 0, MAX_NUM_SURFACES * sizeof(struct cudaExtent));
        (void*)memset(desc, 0, MAX_NUM_SURFACES * sizeof(struct cudaChannelFormatDesc));
        (void*)memset(mipmapDesc, 0, MAX_NUM_SURFACES * sizeof(struct cudaExternalMemoryMipmappedArrayDesc));
        for (uint32_t planeId = 0; planeId < m_bufAttrs[packetIndex].planeCount; planeId++) {
            /* Setting for each plane buffer
             * SP format has 2 planes
             * Planar format has 3 planes  */
            extent[planeId].width = m_bufAttrs[packetIndex].planeWidths[planeId];
            extent[planeId].height = m_bufAttrs[packetIndex].planeHeights[planeId];
            extent[planeId].depth = 1;
            /* For Y */
            if (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_Y8) {
                desc[planeId] = cudaCreateChannelDesc(8, 0, 0, 0,cudaChannelFormatKindUnsigned);
            }
            /* For UV */
            if ((m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_U8V8) ||
                (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_U8_V8) ||
                (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_V8U8) ||
                (m_bufAttrs[packetIndex].planeColorFormats[planeId] == NvSciColor_V8_U8)) {
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
    } else {
        PLOG_ERR("Unsupported layout\n");
        return NVSIPL_STATUS_ERROR;
    }
    if (m_pCudaCopyMem[packetIndex] == nullptr) {
        cudaStatus = cudaHostAlloc((void **)&m_pCudaCopyMem[packetIndex], m_bufAttrs[packetIndex].size, cudaHostAllocDefault);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaHostAlloc");
        PCHK_PTR_AND_RETURN(m_pCudaCopyMem[packetIndex], "m_pCudaCopyMem allocation");
        (void*)memset(m_pCudaCopyMem[packetIndex], 0, m_bufAttrs[packetIndex].size);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::RegisterSignalSyncObj(void)
{
    // Map CUDA Signaler sync objects
    cudaExternalSemaphoreHandleDesc extSemDescSig;
    memset(&extSemDescSig, 0, sizeof(extSemDescSig));
    extSemDescSig.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDescSig.handle.nvSciSyncObj = m_signalSyncObj;
    auto cudaStatus = cudaImportExternalSemaphore(&m_signalerSem, &extSemDescSig);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalSemaphore");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::RegisterWaiterSyncObj(uint32_t index)
{
    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = m_waiterSyncObjs[index];
    auto cudaStatus = cudaImportExternalSemaphore(&m_waiterSem, &extSemDesc);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalSemaphore");

    return NVSIPL_STATUS_OK;
}

//Before calling PreSync, m_nvmBuffers[packetIndex] should already be filled.
SIPLStatus CCudaConsumer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{
    /* Instruct CUDA to wait for the producer fence */
    if( m_FirstCall ) {
        size_t unused;
        auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

        cudaStatus = cudaSetDevice(m_cudaDeviceId);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");
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
    cudaStreamSynchronize(m_streamWaiter);

    uint8_t *second_dst = (uint8_t *)dstptr +
        (size_t)(m_bufAttrs[packetIndex].planeWidths[0U] * m_bufAttrs[packetIndex].planeHeights[0U]);
    cudaStatus = cudaMemcpy2DFromArrayAsync((void *)(second_dst),
                                             (size_t)m_bufAttrs[packetIndex].planeWidths[0U],
                                             *((cudaArray_const_t *)vaddr_plane[1]),
                                             0, 0,
                                             (size_t)m_bufAttrs[packetIndex].planeWidths[0U],
                                             (size_t)(m_bufAttrs[packetIndex].planeHeights[0U]/2),
                                             cudaMemcpyDeviceToDevice,
                                             m_streamWaiter);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2DFromArrayAsync for plane 1");
    cudaStreamSynchronize(m_streamWaiter);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCudaConsumer::ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) {
    uint8_t *plPtr = nullptr;
    SIPLStatus status = NVSIPL_STATUS_OK;
    uint8_t *pBuffer = nullptr;
    size_t uNumBytes = 0U;

    if (m_bufAttrs[packetIndex].layout == NvSciBufImage_BlockLinearType) {
        uNumBytes = (size_t)m_bufAttrs[packetIndex].planeWidths[0]*(int)m_bufAttrs[packetIndex].planeHeights[0]*1.5;

        pBuffer = new (std::nothrow) uint8_t[uNumBytes];
        if (pBuffer == nullptr) {
            PLOG_ERR("Out of memory\n");
            return NVSIPL_STATUS_OUT_OF_MEMORY;
        }

        m_hostBufLen = uNumBytes;
        m_pHostBuf = pBuffer;

        auto cudaStatus = cudaMalloc((void **)&plPtr, m_hostBufLen);
        if (cudaStatus != cudaSuccess) {
            PLOG_ERR("cudaMalloc failed: %u\n", status);
            status = NVSIPL_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        status = BlToPlConvert(packetIndex, (void *)plPtr);
        if (status != NVSIPL_STATUS_OK) {
            PLOG_ERR("BlToPlConvert failed: %u\n", status);
            goto cleanup;
        }
        cudaStatus = cudaMemcpyAsync((void *)m_pHostBuf,
                                     (void *)plPtr,
                                     m_hostBufLen,
                                     cudaMemcpyDeviceToHost,
                                     m_streamWaiter);
        if (cudaStatus != cudaSuccess) {
            PLOG_ERR("cudaMemcpyAsync failed: %u\n", cudaStatus);
            status = NVSIPL_STATUS_ERROR;
            goto cleanup;
        }
        cudaStreamSynchronize(m_streamWaiter);

        PLOG_DBG("ProcessPayload succeed.\n");
        cudaFree(plPtr);

        return status;
    } else {
        PLOG_ERR("Unsupported layout\n");
        return NVSIPL_STATUS_ERROR;
    }

cleanup:
    cudaFree(plPtr);
    if(nullptr != m_pHostBuf) {
        free(m_pHostBuf);
        m_pHostBuf = nullptr;
    }

    return status;
}

SIPLStatus CCudaConsumer::OnProcessPayloadDone(uint32_t packetIndex)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    //dump frames to local file
    if (m_consConfig.bFileDump && (m_frameNum >= DUMP_START_FRAME && m_frameNum <= DUMP_END_FRAME)) {
        if (nullptr != m_pHostBuf && m_hostBufLen > 0) {
            char *buffer = (char *)m_pHostBuf;
            auto ret = fwrite(buffer, 1, m_hostBufLen, m_pOutputFile);
            if(ret != m_hostBufLen) {
                PLOG_ERR("fwrite failed.\n");
                status = NVSIPL_STATUS_ERROR;
                goto cleanup;
            }
        }
    }

cleanup:
    if(nullptr != m_pHostBuf) {
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

