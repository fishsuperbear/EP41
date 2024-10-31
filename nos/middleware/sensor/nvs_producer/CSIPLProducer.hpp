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
#if !NV_IS_SAFETY
//#include "nvmedia_common.h"
#endif
#include "NvSIPLCamera.hpp"
#include "CLateConsumerHelper.hpp"

class CSIPLProducer: public CProducer
{
public:
    CSIPLProducer() = delete;
    CSIPLProducer(NvSciStreamBlock handle, uint32_t uSensor);

    virtual ~CSIPLProducer(void);

    void PreInit(INvSIPLCamera *pCamera, std::shared_ptr<CLateConsumerHelper> lateConsHelper = nullptr);
    virtual SIPLStatus Post(void *pBuffer) override;

protected:
    virtual SIPLStatus HandleClientInit(void) override;
    virtual SIPLStatus HandleStreamInit(void) override;
    virtual SIPLStatus SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList &bufAttrList) override;
    virtual SIPLStatus SetSyncAttrList(PacketElementType userType,
                                       NvSciSyncAttrList &signalerAttrList,
                                       NvSciSyncAttrList &waiterAttrList) override;
    virtual void OnPacketGotten(uint32_t packetIndex) override;
    virtual SIPLStatus RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj) override;
    virtual SIPLStatus RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj) override;
    virtual SIPLStatus HandleSetupComplete(void) override;
    virtual SIPLStatus MapDataBuffer(PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus
    InsertPrefence(PacketElementType userType, uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus CollectWaiterAttrList(uint32_t elementId, std::vector<NvSciSyncAttrList> &unreconciled) override;

#ifdef NVMEDIA_QNX
    virtual bool HasCpuWait(void) {return false;};
#else
    virtual bool HasCpuWait(void) {return true;};
#endif // NVMEDIA_QNX

private:
struct SIPLBuffer
    {
        std::vector<NvSciBufObj> bufObjs;
        std::vector<INvSIPLClient::INvSIPLNvMBuffer *> nvmBuffers;
    };

    SIPLStatus RegisterBuffers(void);
    SIPLStatus SetBufAttrList(PacketElementType userType,
                              INvSIPLClient::ConsumerDesc::OutputType outputType,
                              NvSciBufAttrList &bufAttrList);
    SIPLStatus GetPacketId(std::vector<NvSciBufObj> bufObjs, NvSciBufObj sciBufObj, uint32_t &packetId);
    SIPLStatus MapElemTypeToOutputType(PacketElementType userType, INvSIPLClient::ConsumerDesc::OutputType &outputType);
    SIPLStatus
    GetPostfence(INvSIPLClient::ConsumerDesc::OutputType userType, uint32_t packetIndex, NvSciSyncFence *pPostfence);
    SIPLStatus MapPayload(INvSIPLClient::ConsumerDesc::OutputType userType, void *pBuffer, uint32_t &packetIndex);

private:
    INvSIPLCamera *m_pCamera = nullptr;
    SIPLBuffer m_siplBuffers[MAX_OUTPUTS_PER_SENSOR];
    int32_t m_elemTypeToOutputType[MAX_NUM_ELEMENTS];
    PacketElementType m_outputTypeToElemType[MAX_OUTPUTS_PER_SENSOR];
    std::shared_ptr<CLateConsumerHelper> m_spLateConsHelper = nullptr;
    uint32_t sendImgCount = 0;
};

#endif
