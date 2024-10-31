// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef __CNVCONSUMER__H__
#define __CNVCONSUMER__H__

#include "sensor/nvs_consumer/CConsumer.hpp"
#include "NvSIPLClient.hpp"
//#include "nvmedia_iep.h"
#include "nvmedia_2d.h"
#include "NvSIPLDeviceBlockInfo.hpp"

namespace hozon {
namespace netaos {
namespace desay { 

class CNvMediaConsumer: public CConsumer
{
    public:
        CNvMediaConsumer() = delete;
        CNvMediaConsumer(NvSciStreamBlock handle,
                    uint32_t uSensor,
                    NvSciStreamBlock queueHandle,
                    uint16_t NvMediaWidth,
                    uint16_t NvMediaHeight);
        virtual ~CNvMediaConsumer(void);

    protected:
        virtual SIPLStatus HandleClientInit(void) override;
        virtual SIPLStatus SetDataBufAttrList(NvSciBufAttrList &bufAttrList) override;
        virtual SIPLStatus SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList) override;
        virtual SIPLStatus MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
        virtual SIPLStatus RegisterSignalSyncObj(NvSciSyncObj signalSyncObj) override;
        virtual SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) override;
        virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
        virtual SIPLStatus SetEofSyncObj(void) override;
        virtual SIPLStatus ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
        //virtual bool ToSkipFrame(uint32_t frameNum)override;
        virtual SIPLStatus OnProcessPayloadDone(uint32_t packetIndex) override;
        virtual SIPLStatus UnregisterSyncObjs(void) override;
        virtual bool HasCpuWait(void) {return true;};
        virtual bool ToSkipFrame(uint32_t frameNum) override;
    private:
        SIPLStatus CreateImage(void);
        NvMedia2D* m_pNvMedia = nullptr;
        NvSciBufObj m_pSciBufObjs[MAX_NUM_PACKETS] {nullptr};
        NvSciBufObj outputImage = nullptr;
        NvMedia2DComposeParameters media2DParam;
        FILE *m_pOutputTimeFile = nullptr;
        FILE *m_pOutputImgFile = nullptr;

        uint16_t m_Width;
        uint16_t m_Height;
    };

}
}
}

#endif
