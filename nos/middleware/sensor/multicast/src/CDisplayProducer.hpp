// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CDISPLAY_PRODUCER_HPP
#define CDISPLAY_PRODUCER_HPP

#include <thread>

#include "CProducer.hpp"
#include "nvmedia_2d_sci.h"

class CStitchingConsumer;
class CDisplayProducer : public CProducer
{
  public:
    class BufferInfo
    {
      public:
        BufferInfo()
        {
            for (uint32_t i = 0U; i < MAX_NUM_SENSORS; ++i) {
                m_postfences[i] = NvSciSyncFenceInitializer;
            }
        }

        const NvSciBufObj &GetNvSciBufObj() const
        {
            return m_bufObj;
        }

        const NvSciSyncFence &GetPreFence() const
        {
            return m_prefence;
        }

        ~BufferInfo()
        {
            if (m_bufObj != nullptr) {
                NvSciBufObjFree(m_bufObj);
            }

            NvSciSyncFenceClear(&m_prefence);
            for (uint32_t i = 0U; i < MAX_NUM_SENSORS; ++i) {
                NvSciSyncFenceClear(&m_postfences[i]);
            }
        }

      private:
        NvSciBufObj m_bufObj = nullptr;
        NvSciSyncFence m_prefence = NvSciSyncFenceInitializer; // insert pre-fence from display consumer.
        NvSciSyncFence m_postfences[MAX_NUM_SENSORS];          // insert post-fences from compositors.
        bool m_bIsAvaiable = false;
        friend CDisplayProducer;
    };

  public:
    explicit CDisplayProducer(NvSciStreamBlock handle);
    virtual ~CDisplayProducer();

    void PreInit(uint32_t numPackets, uint32_t width = 1920U, uint32_t height = 1080U);
    virtual SIPLStatus Deinit(void) override;

    void RegisterCompositor(uint32_t uCompositorId, CStitchingConsumer *compositor);
    const NvMediaRect &GetRect(uint32_t uCompositorId) const
    {
        return m_oInputRects[uCompositorId];
    }

    BufferInfo *GetBufferInfo(uint32_t uCompositorId);
    SIPLStatus Submit(uint32_t uCompositorId, const NvSciSyncFence *fence);

  protected:
    virtual SIPLStatus HandleClientInit() override;
    virtual SIPLStatus SetDataBufAttrList(NvSciBufAttrList &bufAttrList) override;
    virtual SIPLStatus SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList) override;
    virtual void OnPacketGotten(uint32_t packetIndex) override;
    virtual SIPLStatus RegisterSignalSyncObj(NvSciSyncObj signalSyncObj) override;
    virtual SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) override;
    virtual SIPLStatus MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj);
    virtual SIPLStatus HandleSetupComplete(void);
    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus GetPostfence(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
    virtual SIPLStatus MapPayload(void *pBuffer, uint32_t &packetIndex) override;
    virtual bool HasCpuWait(void)
    {
        return false;
    }

  private:
    CDisplayProducer(const CDisplayProducer &) = delete;
    CDisplayProducer(CDisplayProducer &&) = delete;
    CDisplayProducer &operator=(const CDisplayProducer &) = delete;
    CDisplayProducer &operator=(CDisplayProducer &&) = delete;

    SIPLStatus ComputeInputRects();
    void DoWork();

    typedef struct
    {
        CStitchingConsumer *compositor = nullptr;
        bool hasSubmitted = false;
    } CompositorInfo;

  private:
    std::vector<uint32_t> m_vCompositorIds;
    CompositorInfo m_compositorInfo[MAX_NUM_SENSORS];
    std::vector<BufferInfo> m_vBufInfo;
    std::vector<uint32_t> m_postBufInfoIds;
    NvSciSyncCpuWaitContext m_displayCPUWaitCtx = nullptr;
    NvSciSyncObj m_cpuSignalSyncObj = nullptr;
    std::unique_ptr<std::thread> m_upthread;
    std::condition_variable m_conditionVar;
    std::mutex m_mutex;

    NvMediaRect m_oInputRects[MAX_NUM_SENSORS];
    uint32_t m_uWidth;
    uint32_t m_uHeight;
    uint32_t m_uCurBufInfoIndex = 0U;
    bool m_bIsRunning = false;
};
#endif
