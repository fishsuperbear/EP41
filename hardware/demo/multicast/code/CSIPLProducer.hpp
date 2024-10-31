// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CSIPLPRODUCER_HPP
#define CSIPLPRODUCER_HPP

#include "CProducer.hpp"

// nvmedia includes
#include "nvmedia_core.h"
#include "NvSIPLCamera.hpp"

class CSIPLProducer: public CProducer
{
public:
    CSIPLProducer() = delete;
    CSIPLProducer(
        NvSciStreamBlock handle,
        uint32_t uSensor,
        INvSIPLCamera* pCamera);
    virtual ~CSIPLProducer(void);

    SIPLStatus Post(INvSIPLClient::INvSIPLNvMBuffer *pBuffer);
protected:
    virtual SIPLStatus HandleClientInit(void) override;
    virtual SIPLStatus CreateBufAttrLists(NvSciBufModule bufModule) override;
    virtual SIPLStatus SetDataBufAttrList(void) override;
    virtual SIPLStatus SetSyncAttrList(void) override;
    virtual void OnPacketGotten(uint32_t packetIndex) override;
    virtual SIPLStatus RegisterSignalSyncObj(void) override;
    virtual SIPLStatus RegisterWaiterSyncObj(uint32_t index) override;
    virtual SIPLStatus HandleSetupComplete(void) override;
    virtual SIPLStatus MapDataBuffer(uint32_t packetIndex) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex) override;
    virtual SIPLStatus InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus GetPostfence(uint32_t packetIndex, NvSciSyncFence *pPostfence) override;
    virtual SIPLStatus MapPayload(void *pBuffer, uint32_t& packetIndex) override;
#ifdef NVMEDIA_QNX
    virtual bool HasCpuWait(void) {return false;};
#else
    virtual bool HasCpuWait(void) {return true;};
#endif // NVMEDIA_QNX

private:
    SIPLStatus RegisterBuffers(void);
    SIPLStatus SetBufAttrList(INvSIPLClient::ConsumerDesc::OutputType outputType,
        NvSciBufAttrList& bufAttrList);

    INvSIPLCamera* m_pCamera;
    INvSIPLClient::ConsumerDesc::OutputType m_ispOutputType;
    NvSciBufAttrList m_rawBufAttrList = nullptr;
    std::vector<NvSciBufObj> m_vIspBufObjs;
    std::vector<NvSciBufObj> m_vRawBufObjs;
    INvSIPLClient::INvSIPLNvMBuffer *m_nvmBuffers[MAX_NUM_PACKETS] {nullptr};
    /* Virtual address for the meta buffer */
    MetaData* m_metaPtrs[MAX_NUM_PACKETS];
};
#endif
