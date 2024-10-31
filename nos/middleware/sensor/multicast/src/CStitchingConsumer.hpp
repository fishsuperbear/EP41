// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CCOMPOSITE_CONSUMER_HPP
#define CCOMPOSITE_CONSUMER_HPP

#include "nvmedia_2d_sci.h"

#include "CConsumer.hpp"
#include "CDisplayProducer.hpp"

class CStitchingConsumer : public CConsumer
{
  public:
    CStitchingConsumer() = delete;
    CStitchingConsumer(NvSciStreamBlock handle, uint32_t uSensorId, NvSciStreamBlock queueHandle);

    virtual ~CStitchingConsumer(void);

    void PreInit(std::shared_ptr<CDisplayProducer> &spDisplayProd);
    SIPLStatus Deinit(void) override;

    const NvSciBufAttrList &GetBufferAttributList()
    {
        return m_dataBufAttrList;
    }
    SIPLStatus RegisterDstNvSciObjBuf(NvSciBufObj obj);

  protected:
    virtual SIPLStatus HandleClientInit() override;
    virtual SIPLStatus SetDataBufAttrList(NvSciBufAttrList &bufAttrList) override;
    virtual SIPLStatus SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList) override;
    virtual SIPLStatus MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus RegisterSignalSyncObj(NvSciSyncObj signalSyncObj) override;
    virtual SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) override;
    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus SetEofSyncObj(void) override;
    virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
    virtual SIPLStatus HandlePayload(void) override;
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
    virtual bool HasCpuWait(void)
    {
        return true;
    }

    SIPLStatus UnRegisterDstNvSciObjBuf(NvSciBufObj obj);

    struct DestroyNvMedia2DDevice
    {
        void operator()(NvMedia2D *p) const
        {
            NvMedia2DDestroy(p);
        }
    };

  private:
    SIPLStatus Set2DParameter(uint32_t packetId, NvSciBufObj bufObj);

  private:
    std::unique_ptr<NvMedia2D, DestroyNvMedia2DDevice> m_up2DDevice;
    std::shared_ptr<CDisplayProducer> m_spDisplayProd;
    NvSciBufObj m_srcBufObjs[MAX_NUM_PACKETS];
    std::vector<NvSciBufObj> m_dstBufObjs;
    NvSciSyncObj m_2DSignalSyncObj = nullptr;
    NvMedia2DComposeParameters m_params;
    NvSciBufAttrList m_dataBufAttrList = nullptr;
};
#endif
