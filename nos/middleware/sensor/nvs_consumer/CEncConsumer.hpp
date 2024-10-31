// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CENCCONSUMER_H
#define CENCCONSUMER_H

#include "sensor/nvs_consumer/CConsumer.hpp"
#include "NvSIPLClient.hpp"
#include "nvmedia_iep.h"
#include "nvmedia_2d.h"
#include "NvSIPLDeviceBlockInfo.hpp"
#include "sensor/nvs_consumer/CEncManager.h"

namespace hozon {
namespace netaos {
namespace desay { 

class CEncConsumer: public CConsumer
{
    public:
        CEncConsumer() = delete;
        CEncConsumer(NvSciStreamBlock handle,
                          uint32_t uSensor,
                          NvSciStreamBlock queueHandle,
                          uint16_t encodeWidth,
                          uint16_t encodeHeight);
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
        virtual bool HasCpuWait(void) {return true;};
        virtual SIPLStatus OnDataBufAttrListReceived(NvSciBufAttrList bufAttrList) override;
        virtual bool ToSkipFrame(uint32_t frameNum) override;

    private:
        struct DestroyNvMediaIEP
        {
            void operator ()(NvMediaIEP *p) const
            {
                NvMediaIEPDestroy(p);
            }
        };
        struct DestroyNvMedia2D
        {
            void operator ()(NvMedia2D *p) const
            {
                NvMedia2DDestroy(p);
            }
        };
        SIPLStatus InitEncoder(NvSciBufAttrList bufAttrList);
        SIPLStatus EncodeOneFrame(NvSciBufObj pSciBufObj,
                                  uint8_t **ppOutputBuffer,
                                  size_t *pNumBytes,
                                  NvSciSyncFence *pPostfence);
        SIPLStatus SetEncodeConfig(void);
        SIPLStatus InitImage2DAndEncoder();
        SIPLStatus AllocateIEPEofSyncObj(NvSciSyncObj *syncObj, NvSciSyncModule syncModule, NvMediaIEP *handle);
        SIPLStatus ConvertToBL(NvSciBufObj pSciBufObjPL, NvSciBufObj pSciBufObjBL, NvSciSyncFence* pPostfence);

        std::unique_ptr<NvMediaIEP, DestroyNvMediaIEP> m_pNvMIEP {nullptr};
        std::unique_ptr<NvMedia2D, DestroyNvMedia2D> m_pNvMedia2D {nullptr};
        NvSciBufObj m_pSciBufObjs[MAX_NUM_PACKETS] {nullptr};
        // NvSciBufObj m_pSciBufObjsBL[MAX_NUM_PACKETS] {nullptr};
        NvSciBufObj m_pBLBufObj = nullptr;
        // BufferAttrs m_bufAttrs[MAX_NUM_PACKETS];
        FILE *m_pOutputFile = nullptr;
        FILE *m_pOutputTimeFile = nullptr;
        NvMediaEncodeConfigH264 m_stEncodeConfigH264Params;
        NvMediaEncodeConfigH265 m_stEncodeConfigH265Params;
        uint16_t m_encodeWidth = 0;
        uint16_t m_encodeHeight = 0;
        uint8_t *m_pEncodedBuf = nullptr;
        size_t m_encodedBytes = 0;
        uint32_t m_frameType = 0;
        uint32_t m_codecType = 0;
        bool m_uhpMode = false;
        int32_t m_srcLayout = 0;
        NvSciBufAttrList m_inputSciBufAttrList = nullptr;
        NvSciBufAttrList m_blSciBufAttrList = nullptr;
        NvMedia2DComposeParameters m_media2DParam {0};
        NvSciSyncCpuWaitContext m_waitIepCtx = nullptr;
        NvSciSyncObj m_IEPSignalSyncObj = nullptr;
        NvSciSyncObj m_signalSyncObj = nullptr;
        uint32_t m_frameSampling = 0;
        uint32_t m_framesInSec = 0;
        uint64_t m_curSec = 0;
        uint64_t m_samplingNanoDur = 0;
        bool m_curFrameSkipped = false;
        Multicast_EncodedImage encoded_image_ {0};
        uint64_t seq_ = 0;
    };

}
}
}

#endif
