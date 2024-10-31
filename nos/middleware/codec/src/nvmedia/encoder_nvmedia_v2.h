/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef CODEC_ENCODER_NVMEDIA_V2_H_
#define CODEC_ENCODER_NVMEDIA_V2_H_
#pragma once

#include "codec/include/encoder.h"
#include "codec/src/nvmedia/CUtils.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvmedia_2d.h"
#include "nvmedia_iep.h"

namespace hozon {
namespace netaos {
namespace codec {

#define MAX_NUM_PACKETS 6

class EncoderNvmediaV2 : public Encoder {
   public:
    EncoderNvmediaV2() = default;
    ~EncoderNvmediaV2();
    /**
   * @brief 初始化decode
   *
   * @return true 初始化成功返回
   * @return false 初始化失败
   */
    CodecErrc Init(const std::string& config_file);
    CodecErrc Init(const EncodeInitParam& init_param);

    /**
   * @brief 解码H265数据并输出mat
   *
   * @param in_message yuv420sp(nv12)
   * @param out_image 输出的编码后H265流数据
   * @return 0 编码成功返回
   * @return <0> 编码失败返回
   */
    CodecErrc Process(const std::vector<std::uint8_t>& in_buff, std::vector<std::uint8_t>& out_buff,
                      FrameType& frame_type) override;
    CodecErrc Process(const std::string& in_buff, std::string& out_buff, FrameType& frame_type) override;
    CodecErrc Process(const void* in, std::vector<std::uint8_t>& out_buff, FrameType& frame_type) override;
    CodecErrc Process(const void* in, std::string& out_buff, FrameType& frame_type) override;

    CodecErrc GetEncoderParam(int param_type, void** param);
    CodecErrc SetEncoderParam(int param_type, void* param);

   private:
    void* context_;

   protected:
    int GetBufAttrs(NvSciBufAttrList& bufAttrList);
    int GetSignalerAttrs(NvSciSyncAttrList& attr_list);
    int GetWaiterAttrs(NvSciSyncAttrList& attr_list);
    int RegisterBufObj(NvSciBufObj bufObj);
    int RegisterSignalerSyncObj(NvSciSyncObj signalSyncObj);
    int RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj);
    int SetPrefence(NvSciSyncFence& prefence);
    int SetEofSyncObj(void);
    int ProcessPayload(uint32_t packetIndex, NvSciSyncFence* pPostfence);
    int UnregisterSyncObjs(void);

    // int OnProcessPayloadDone(uint32_t packetIndex);
    bool HasCpuWait(void) { return true; };

    int SetBufAttrs(NvSciBufAttrList bufAttrListt);

    bool ConvertRgbToScibuf(void* rgb_device, NvSciBufObj* bufObj);
    bool MapNvScibufToCuda(const NvSciBufObj scibuf);

   private:
    struct DestroyNvMediaIEP {
        void operator()(NvMediaIEP* p) const { NvMediaIEPDestroy(p); }
    };

    struct DestroyNvMedia2D {
        void operator()(NvMedia2D* p) const { NvMedia2DDestroy(p); }
    };

    int InitEncoder(NvSciBufAttrList bufAttrList);
    CodecErrc EncodeOneFrame(NvSciBufObj pSciBufObj, std::vector<uint8_t>& out, NvSciSyncFence* pPostfence);
    CodecErrc EncodeOneFrame(NvSciBufObj pSciBufObj, std::string& out);
    int SetEncodeConfig(void);
    int InitImage2DAndEncoder();
    int AllocateIEPEofSyncObj(NvSciSyncObj* syncObj, NvSciSyncModule syncModule, NvMediaIEP* handle);
    CodecErrc ConvertToBL(NvSciBufObj pSciBufObjPL, NvSciBufObj pSciBufObjBL, NvSciSyncFence* pPostfence);
    int InitCudaAndEncoder();
    SIPLStatus BlToPlConvert(void* dstptr);

    std::unique_ptr<NvMediaIEP, DestroyNvMediaIEP> m_pNvMIEP{nullptr};
    std::unique_ptr<NvMedia2D, DestroyNvMedia2D> m_pNvMedia2D{nullptr};
    NvSciBufObj m_pSciBufObjs[MAX_NUM_PACKETS]{nullptr};
    NvSciBufObj m_pBLBufObj = nullptr;
    NvSciSyncObj m_iepSignalSyncObj = nullptr;
    NvSciSyncObj m_media2DSignalSyncObj = nullptr;
    FILE* m_pOutputFile = nullptr;
    NvMediaEncodeConfigH264 m_stEncodeConfigH264Params{};
    NvMediaEncodeConfigH265 m_stEncodeConfigH265Params{};
    uint8_t* m_pEncodedBuf = nullptr;
    size_t m_encodedBytes = 0;
    uint32_t m_frameType = 0;

    bool m_uhpMode = false;

    NvMedia2DComposeParameters m_media2DParam{0};

    NvSciSyncObj m_waiterSyncObj = nullptr;
    NvSciBufModule m_sciBufModule = nullptr;
    NvSciSyncModule m_sciSyncModule = nullptr;
    NvSciSyncCpuWaitContext m_waitIepCtx = nullptr;
    int32_t encoder_state_ = 0;

    EncodeInitParam init_param_{0};
    NvSciRmGpuId gpu_id_{0};
    cudaStream_t cuda_stream_ = nullptr;

    // for cuda rgb to nv12
    struct CudaData {
        void* cuda_ptr = nullptr;
        cudaExternalMemory_t ext_mem = nullptr;
        cudaMipmappedArray_t mipmap_array[3] = {0};
        cudaArray_t mipLevel_array[3] = {0};
        // cudaExternalSemaphore_t signaler_sem = nullptr;
        // cudaExternalSemaphore_t waiter_sem = nullptr;
        BufferAttrs bufAttrs{};
        cudaSurfaceObject_t surface_luma;
        cudaSurfaceObject_t surface_chroma;

    } mapping_cuda_data_;

    uint32_t encode_gop_ = 0;

    friend class CLateConsumerHelper;
};
}  // namespace codec
}  // namespace netaos
}  // namespace hozon
#endif