/**
 * @file CVirtualCamProducer.hpp
 * @author zax (maxiaotian@hozonauto.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * Copyright Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * 
 */
#ifndef MIDDLEWARE_SENSOR_NVS_PRODUCER_CVIRTUALCAMPRODUCER_HPP_
#define MIDDLEWARE_SENSOR_NVS_PRODUCER_CVIRTUALCAMPRODUCER_HPP_

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
#include "CLateConsumerHelper.hpp"
#include "nv_replayer_api.h"
#include "nvmedia_core.h"
#include "nvmedia_ide.h"
#include "nvmedia_parser.h"

using hozon::netaos::codec::InputBuf;
using hozon::netaos::codec::PicInfo;

class BufferPool {
   public:
    typedef struct {
        NvSciBufObj image;
        std::atomic<uint8_t> ref_count;
    } ImageBuf;

    explicit BufferPool(uint16_t capacity, uint16_t sid) : capacity_(capacity) {
        //
        head_ = (ImageBuf*)calloc(1, sizeof(ImageBuf) * capacity_);
        LOG_INFO("sid=%d, init BufferPool\n", sid);
        is_full_ = false;
    }

    ~BufferPool() {
        for (int i = 0; i < capacity_; ++i) {
            NvSciBufObjFree(head_[i].image);
        }

        delete head_;
    }

    bool MapBuffer(uint16_t index, const NvSciBufObj& obj) {
        auto ret = NvSciBufObjDup(obj, &head_[index].image);
        if (ret != NvSciError_Success) {
            return false;
        }
        // head_[index].image = obj;
        head_[index].ref_count = 0;
        // LOG_INFO("set [%d] imagep=%p, inputp=%p\n", index, head_[index].image, obj);
        LOG_INFO("head p=%p\n", head_);

        return true;
    }

    bool AquireBuffer(uint16_t* index, NvSciBufObj* image) {
        std::unique_lock<std::mutex> lock(mtx_);
        // if full , wait buffer release.
        int i = 0;
        // cv_.wait(lock, [this] { return !is_full_; });
        for (; i < capacity_; ++i) {
            if (!head_[i].ref_count++) {
                *image = head_[i].image;
                *index = i;
                // LOG_INFO("get [%d] imagep=%p, inputp=%p\n", i, head_[i].image, *image);
                return true;
                break;
            }
        }
        if (i == capacity_ - 1) {
            is_full_ = true;
        }
        return false;
    }

    bool ReleaseBuffer(const NvSciBufObj& image) {
        std::unique_lock<std::mutex> lock(mtx_);
        for (int i = 0; i < capacity_; ++i) {
            if (head_[i].image == image) {
                if (head_[i].ref_count != 0) {
                    head_[i].ref_count = 0;
                    // cv_.notify_one();
                    return true;
                }
            }
        }
        return false;
    }

    bool ReleaseBuffer(uint16_t index) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (head_[index].ref_count != 0) {
            head_[index].ref_count = 0;
            is_full_ = false;
            // cv_.notify_one();
            return true;
        }
        return false;
    }

   private:
    ImageBuf* head_ = nullptr;
    uint16_t capacity_ = 0;
    std::mutex mtx_;
    std::atomic<bool> is_full_;
    std::condition_variable cv_;
};

class CVirtualCamProducer : public CProducer {
   public:
    CVirtualCamProducer() = delete;
    CVirtualCamProducer(NvSciStreamBlock handle, PicInfo* pic_info);
    virtual ~CVirtualCamProducer(void);
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
    SIPLStatus GetPostfence(uint32_t packetIndex, NvSciSyncFence* pPostfence) override;
    SIPLStatus MapPayload(void* pBuffer, uint32_t& packetIndex) override;

    SIPLStatus GetIdeImageAttributes(const NvSciBufAttrList& imageAttr);
    SIPLStatus DecodeOnceSync(const InputBuf& src_data, NvSciBufObj out);

    virtual bool HasCpuWait(void) { return true; };

   private:
    static NvMediaStatus ParserCb_DecodePicture(void* ctx, NvMediaParserPictureData* pic_data);
    static int32_t ParserCb_BeginSequence(void* ctx, const NvMediaParserSeqInfo* pnvsi);
    static NvMediaStatus ParserCb_DisplayPicture(void* ctx, NvMediaRefSurface* p_ref_surf, int64_t llpts);
    static void ParserCb_UnhandledNALU(void* ctx, const uint8_t* buf, int32_t size);
    static NvMediaStatus ParserCb_AllocPictureBuffer(void* ctx, NvMediaRefSurface** pp_ref_surf);
    static void ParserCb_Release(void* ctx, NvMediaRefSurface* p_ref_surf);
    static void ParserCb_AddRef(void* ctx, NvMediaRefSurface* p_ref_surf);
    static NvMediaStatus ParserCb_GetBackwardUpdates(void* ptr, NvMediaVP9BackwardUpdates* backwardUpdate);

    PacketElementType m_outputTypeToElemType[MAX_OUTPUTS_PER_SENSOR];
    std::shared_ptr<CLateConsumerHelper> m_spLateConsHelper = nullptr;
    uint32_t sendImgCount;

    // std::vector<NvSciBufObj> m_vIspBufObjs;
    // MetaData* m_metaPtrs[MAX_NUM_PACKETS];
    PicInfo* pic_info_ = nullptr;
    NvMediaIDE* decoder_ = nullptr;
    unique_ptr<BufferPool> buffer_pool_ = nullptr;

    std::atomic_bool is_setup_complete_{false};
    NvMediaParser* media_parser_ = nullptr;
    bool pic_info_h265_parsed_ = false;
    NvMediaPictureInfoH265 pic_info_h265_{0};
    std::mutex pic_info_h265_mutex_;
    std::condition_variable pic_info_h265_condition_;
    NvSciBufObj last_sci_buf_ = nullptr;
    std::string last_buf_;

    std::unique_ptr<std::ofstream> file_;
};
#endif  // MIDDLEWARE_SENSOR_NVS_PRODUCER_CVIRTUALCAMPRODUCER_HPP_
