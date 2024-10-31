// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CDISPLAY_CONSUMER_HPP
#define CDISPLAY_CONSUMER_HPP

#include "CConsumer.hpp"
#include "COpenWFDController.hpp"

namespace hozon {
namespace netaos {
namespace desay {

class CDisplayConsumer : public CConsumer
{
  public:
    CDisplayConsumer() = delete;
    CDisplayConsumer(NvSciStreamBlock handle, uint32_t uSensorId, NvSciStreamBlock queueHandle);

    void PreInit(const std::shared_ptr<COpenWFDController> &wfdController, uint32_t wfdPipelineId = 0U);
    virtual ~CDisplayConsumer(void);

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
    virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
    virtual SIPLStatus HandleSetupComplete(void);
    virtual bool HasCpuWait(void)
    {
        return false;
    }

  private:
    std::shared_ptr<COpenWFDController> m_spWFDController;
    WFDint m_wfdPipelineId;

    BufferAttrs m_bufAttrs[MAX_NUM_PACKETS];
    bool m_inited;
};

}
}
}
#endif
