// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CCUDACONSUMER_HPP
#define CCUDACONSUMER_HPP

#include "hw_nvmedia_eventhandler_common_impl.h"

#include "CConsumer.hpp"

// cuda includes
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "hw_nvmedia_gpu_common.hpp"

class CCudaConsumer: public CConsumer
{
public:
    CCudaConsumer() = delete;
    CCudaConsumer(NvSciStreamBlock handle, u32 uSensor, u32 i_blockindex, u32 i_sensorindex, NvSciStreamBlock queueHandle, u32 i_capturewidth, u32 i_captureheight);
    virtual ~CCudaConsumer(void);
    static SIPLStatus GetBufAttrList(NvSciBufAttrList outBufAttrList);
    static SIPLStatus GetSyncWaiterAttrList(NvSciSyncAttrList outWaiterAttrList);
    virtual SIPLStatus HandleSetupComplete(void)
        {
            m_streamPhase = StreamPhase_Streaming;
            // cuInit(0);
            // cuCtxPushCurrent(_cuda_context);
            return NVSIPL_STATUS_OK;
        }
public:
    virtual hw_ret_s32 RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
        HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig);
protected:
    virtual SIPLStatus HandleClientInit(void) override;
    virtual SIPLStatus SetDataBufAttrList(NvSciBufAttrList &bufAttrList) override;
    virtual SIPLStatus SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList) override;
    virtual SIPLStatus MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus RegisterSignalSyncObj(NvSciSyncObj signalSyncObj) override;
    virtual SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) override;
    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
    virtual SIPLStatus UnregisterSyncObjs(void) override;
    virtual bool HasCpuWait(void)
    {
        return true;
    }

private:
    SIPLStatus InitCuda(void);
    SIPLStatus BlToPlConvert(uint32_t packetIndex, void *dstptr);

    int m_cudaDeviceId = 0;
    uint8_t *m_pCudaCopyMem[MAX_NUM_PACKETS];
    void *m_devPtr[MAX_NUM_PACKETS];
    cudaExternalMemory_t m_extMem[MAX_NUM_PACKETS];
    cudaStream_t m_streamWaiter = nullptr;
    cudaExternalSemaphore_t m_signalerSem;
    cudaExternalSemaphore_t m_waiterSem;
    BufferAttrs m_bufAttrs[MAX_NUM_PACKETS] = {};
    cudaMipmappedArray_t m_mipmapArray[MAX_NUM_PACKETS][MAX_NUM_SURFACES] = {};
    cudaArray_t m_mipLevelArray[MAX_NUM_PACKETS][MAX_NUM_SURFACES] = {};

    FILE *m_pOutputFile = nullptr;
    uint8_t *m_pHostBuf = nullptr;
    size_t m_hostBufLen;
    bool m_FirstCall;
    u32                 _testframecount = 0;
    u32                 _blockindex = 0;
    u32                 _sensorindex = 0;
    INvSIPLClient::ConsumerDesc::OutputType     _outputtype;
    hw_video_sensorpipelinecudadatacb   _datacb = nullptr;
    //hw_nvmedia_cuda_context_t* _pgpuimage = nullptr;       //hw_nvmedia_cuda_context_t
    struct hw_video_sensorpipelinecudasdatacbconfig_t	_cudaconfig;
    // 0 or 1, 1 means has log that the _datacb is nullptr tip log
    u32                             _blogdatacbnull = 0;
    GPUImage                        _gpuimage;
    // byte count of data of GPUImage _gpuimage
    u32                             _buffsize;
    GPUImage*                       _pgpuimage = nullptr;
    void*                           _pcontext = nullptr;
private:
    // will be release when ~CCudaConsumer
    uint8_t*                        _plPtr = nullptr;
    // will be release when ~CCudaConsumer
    void*                           _pimage = nullptr;
private:
    HW_VIDEO_REGCUDADATACB_IMGTYPE  _imgtype;
    HW_VIDEO_REGCUDADATACB_INTERPOLATION    _interpolation;
    u32                             _capturewidth;
    u32                             _captureheight;
    // set by register cuda data callback
    u32                             _width;
    // set by register cuda data callback
    u32                             _height;
    u32                             _busecaptureframerate;
    // valid when _busecaptureframerate is 0
    u32                             _customframerate;
    friend class CAttributeProvider;
    CUcontext                       _cuda_context;
};
#endif
