// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CENCCONSUMER_H
#define CENCCONSUMER_H

#include "CConsumer.hpp"
#include "NvSIPLClient.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"
#include "nvmedia_iep.h"

class CEncConsumer : public CConsumer
{
  public:
    CEncConsumer() = delete;
    CEncConsumer(NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle);
    virtual ~CEncConsumer(void);

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
    virtual SIPLStatus UnregisterSyncObjs(void) override;
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
    virtual bool HasCpuWait(void)
    {
        return true;
    };
    virtual SIPLStatus OnDataBufAttrListReceived(NvSciBufAttrList bufAttrListt) override;

  private:
    struct DestroyNvMediaIEP
    {
        void operator()(NvMediaIEP *p) const
        {
            NvMediaIEPDestroy(p);
        }
    };
    SIPLStatus InitEncoder(NvSciBufAttrList bufAttrList);
    SIPLStatus
    EncodeOneFrame(NvSciBufObj pSciBufObj, uint8_t **ppOutputBuffer, size_t *pNumBytes, NvSciSyncFence *pPostfence);
    SIPLStatus SetEncodeConfig(void);

    std::unique_ptr<NvMediaIEP, DestroyNvMediaIEP> m_pNvMIEP{ nullptr };
    NvSciBufObj m_pSciBufObjs[MAX_NUM_PACKETS]{ nullptr };
    NvSciSyncObj m_IEPSignalSyncObj = nullptr;
    FILE *m_pOutputFile = nullptr;
    NvMediaEncodeConfigH264 m_stEncodeConfigH264Params{};
    uint8_t *m_pEncodedBuf = nullptr;
    size_t m_encodedBytes = 0;

    friend class CLateConsumerHelper;
};
#endif
