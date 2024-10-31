/**
 * @file CVirtualCamVicProducer.hpp
 * @author zax (maxiaotian@hozonauto.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * Copyright Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * 
 */
#ifndef MIDDLEWARE_SENSOR_NVS_PRODUCER_CVirtualCamVicProducer_HPP_
#define MIDDLEWARE_SENSOR_NVS_PRODUCER_CVirtualCamVicProducer_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include "CProducer.hpp"

// nvmedia includes
#include "nv_replayer_api.h"
#include "nvmedia_core.h"
#include "nvmedia_ide.h"
// image2d includes
#include "CLateConsumerHelper.hpp"
#include "nvmedia_2d.h"
#include "nvmedia_2d_sci.h"
#include "nvmedia_core.h"
#include "nvmedia_parser.h"

using hozon::netaos::codec::InputBuf;
using hozon::netaos::codec::PicInfo;

class CVirtualCamVicProducer : public CProducer {
   public:
    CVirtualCamVicProducer() = delete;
    CVirtualCamVicProducer(NvSciStreamBlock handle, PicInfo* pic_info);
    virtual ~CVirtualCamVicProducer(void);
    bool IsComplete();
    void PreInit(std::shared_ptr<CLateConsumerHelper> lateConsHelper = nullptr);

   protected:
    virtual SIPLStatus HandleClientInit(void) override;
    virtual SIPLStatus HandleStreamInit(void) override;
    virtual SIPLStatus SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList& bufAttrList) override;
    virtual SIPLStatus SetSyncAttrList(PacketElementType userType, NvSciSyncAttrList& signalerAttrList, NvSciSyncAttrList& waiterAttrList) override;
    virtual void OnPacketGotten(uint32_t packetIndex) override;
    virtual SIPLStatus RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj) override;
    virtual SIPLStatus RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj) override;
    virtual SIPLStatus HandleSetupComplete(void) override;
    virtual SIPLStatus MapDataBuffer(PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus InsertPrefence(PacketElementType userType, uint32_t packetIndex, NvSciSyncFence& prefence) override;
    virtual SIPLStatus CollectWaiterAttrList(uint32_t elementId, std::vector<NvSciSyncAttrList>& unreconciled) override;
    SIPLStatus RegisterBuffers(void);
    SIPLStatus SetBufAttrList(NvSciBufAttrList& bufAttrList);
    SIPLStatus GetPacketId(std::vector<NvSciBufObj> bufObjs, NvSciBufObj sciBufObj, uint32_t& packetId);
    SIPLStatus MapElemTypeToOutputType(PacketElementType userType, INvSIPLClient::ConsumerDesc::OutputType& outputType);
    SIPLStatus GetPostfence(uint32_t packetIndex, NvSciSyncFence* pPostfence) override;
    SIPLStatus MapPayload(void* pBuffer, uint32_t& packetIndex) override;

    SIPLStatus GetIdeImageAttributes(const NvSciBufAttrList& imageAttr);
    SIPLStatus DecodeOnceSync(const InputBuf& src_data, NvSciBufObj out);

    virtual bool HasCpuWait(void) { return true; };

    SIPLStatus ReconcileAndAllocBuffers(NvSciBufAttrList& bufAttrList1, NvSciBufAttrList& bufAttrList2, std::vector<NvSciBufObj>& vBufObjs);
    SIPLStatus FillMedia2DOutputBufAttrList(NvSciBufAttrList sciBufAttrList);

   private:
    struct DestroyNvMedia2D {
        void operator()(NvMedia2D* handle) const {
            if (handle) {
                NvMediaStatus result = NvMedia2DDestroy(handle);
                if (result == NVMEDIA_STATUS_OK) {
                    handle = nullptr;
                } else {
                    LOG_ERR("NvMedia2DDestroy failed with %d\n", result);
                }
            }
        }
    };

    int32_t m_elemTypeToOutputType[MAX_NUM_ELEMENTS];
    PacketElementType m_outputTypeToElemType[MAX_OUTPUTS_PER_SENSOR];
    std::shared_ptr<CLateConsumerHelper> m_spLateConsHelper = nullptr;
    uint32_t sendImgCount;
    static NvMediaStatus ParserCb_DecodePicture(void* ctx, NvMediaParserPictureData* pic_data);
    static int32_t ParserCb_BeginSequence(void* ctx, const NvMediaParserSeqInfo* pnvsi);
    static NvMediaStatus ParserCb_DisplayPicture(void* ctx, NvMediaRefSurface* p_ref_surf, int64_t llpts);
    static void ParserCb_UnhandledNALU(void* ctx, const uint8_t* buf, int32_t size);
    static NvMediaStatus ParserCb_AllocPictureBuffer(void* ctx, NvMediaRefSurface** pp_ref_surf);
    static void ParserCb_Release(void* ctx, NvMediaRefSurface* p_ref_surf);
    static void ParserCb_AddRef(void* ctx, NvMediaRefSurface* p_ref_surf);
    static NvMediaStatus ParserCb_GetBackwardUpdates(void* ptr, NvMediaVP9BackwardUpdates* backwardUpdate);

    // std::vector<NvSciBufObj> m_vIspBufObjs;
    // MetaData* m_metaPtrs[MAX_NUM_PACKETS];
    PicInfo* pic_info_ = nullptr;
    NvMediaIDE* decoder_ = nullptr;
    std::vector<NvSciBufObj> ide_output_;
    std::vector<NvSciBufObj> vic_output_;
    std::unique_ptr<std::thread> post_thread_;
    std::atomic_bool is_setup_complete_{false};
    NvMediaParser* media_parser_ = nullptr;
    bool pic_info_h265_parsed_ = false;
    NvMediaPictureInfoH265 pic_info_h265_{0};
    std::mutex pic_info_h265_mutex_;
    std::condition_variable pic_info_h265_condition_;
    NvSciBufObj last_sci_buf_ = nullptr;
    std::string last_buf_;

    unique_ptr<NvMedia2D, DestroyNvMedia2D> nvmedia2d_;
    NvMedia2DComposeParameters nvmedia2d_params_;

    NvSciBufAttrList ide_buf_attrlist_ = nullptr;
    NvSciSyncAttrList ide_signaler_attrlist_ = nullptr;
    NvSciSyncObj ide_signal_syncobj_;
    NvSciSyncCpuWaitContext ide_cpu_wait_;
    // TODO(mxt):
    std::vector<int> send_flag_;
    std::mutex mtx_;

    FILE* file = nullptr;
    std::unique_ptr<std::ofstream> file_;
};
#endif  // MIDDLEWARE_SENSOR_NVS_PRODUCER_CVirtualCamVicProducer_HPP_
