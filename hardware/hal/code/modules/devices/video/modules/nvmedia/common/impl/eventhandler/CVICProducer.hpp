// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CVICPRODUCER_HPP
#define CVICPRODUCER_HPP

#include "CProducer.hpp"

// nvmedia includes
#include "nvmedia_core.h"

class ICascadedProvider;
class CBuffer;

class CVICProducer: public CProducer
{
public:
    CVICProducer( NvSciStreamBlock handle, uint32_t uSensor, ICascadedProvider* pCascadedProvider );
    virtual ~CVICProducer();
    virtual SIPLStatus Post(void *pBuffer) override;

protected:
    virtual SIPLStatus HandleClientInit() override;
    virtual SIPLStatus SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList &bufAttrList) override;
    virtual SIPLStatus SetSyncAttrList(PacketElementType userType,
                                       NvSciSyncAttrList &signalerAttrList,
                                       NvSciSyncAttrList &waiterAttrList) override;
    virtual void OnPacketGotten( uint32_t packetIndex ) override;
    virtual SIPLStatus RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj) override;
    virtual SIPLStatus RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj) override;
    virtual SIPLStatus HandleSetupComplete() override;
    virtual SIPLStatus MapDataBuffer(PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) override;
    virtual SIPLStatus
    InsertPrefence(PacketElementType userType, uint32_t packetIndex, NvSciSyncFence &prefence) override;
    virtual SIPLStatus MapPayload(void *pBuffer, uint32_t &packetIndex) override;
#ifdef NVMEDIA_QNX
    virtual bool HasCpuWait() { return false; }
#else
    virtual bool HasCpuWait() { return true; }
#endif // NVMEDIA_QNX

private:
    struct SIPLBuffer
    {
        std::vector<NvSciBufObj> bufObjs;
        std::vector<INvSIPLClient::INvSIPLNvMBuffer *> nvmBuffers;
    };
    SIPLStatus SetBufAttrList(PacketElementType userType,
                              INvSIPLClient::ConsumerDesc::OutputType outputType,
                              NvSciBufAttrList &bufAttrList);
    SIPLStatus GetPacketId(std::vector<NvSciBufObj> bufObjs, NvSciBufObj sciBufObj, uint32_t &packetId);
    SIPLStatus MapElemTypeToOutputType(PacketElementType userType, INvSIPLClient::ConsumerDesc::OutputType &outputType);
    SIPLStatus
    GetPostfence(INvSIPLClient::ConsumerDesc::OutputType userType, uint32_t packetIndex, NvSciSyncFence *pPostfence);
private:
    ICascadedProvider*       m_pCascadedProvider;
    std::vector<NvSciBufObj> m_vBufObjs;
    CBuffer*                 m_pBuffers[MAX_NUM_PACKETS] { nullptr };
};
#endif
