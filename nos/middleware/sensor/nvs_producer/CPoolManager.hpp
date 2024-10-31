// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CPOOLMANAGER_HPP
#define CPOOLMANAGER_HPP

#include "CEventHandler.hpp"
#include "CLateConsumerHelper.hpp"
#include "CUtils.hpp"
#include "Common.hpp"
#include "nvscistream.h"

// Define attributes of a packet element.
struct ElemAttr {
    uint32_t userName = 0U; /* The application's name for the element */
    NvSciBufAttrList bufAttrList = nullptr;

    ~ElemAttr() {
        if (bufAttrList != nullptr) {
            NvSciBufAttrListFree(bufAttrList);
            bufAttrList = nullptr;
        }
    }
};

class CPoolManager : public CEventHandler {
   public:
    CPoolManager(NvSciStreamBlock handle, uint32_t uSensor, uint32_t numPackets, bool isC2C);
    ~CPoolManager(void);

    void PreInit(std::shared_ptr<CLateConsumerHelper> lateConsHelper = nullptr);
    SIPLStatus Init();
    virtual EventStatus HandleEvents(void) override;
    void SetElemTypesToSkip(const vector<uint32_t>& vElemTypesToSkip);

   private:
    SIPLStatus HandlePoolBufferSetup(void);
    SIPLStatus HandlePacketsStatus(void);
    SIPLStatus HandleElements(void);
    SIPLStatus HandleC2CElements(void);
    SIPLStatus HandleBuffers(void);
    void FreeElements(void);

    uint32_t m_numConsumers = 0U;
    // Producer packet element attributue
    uint32_t m_numProdElem = 0U;
    ElemAttr m_prodElems[MAX_NUM_ELEMENTS];

    // Consumer packet element attributue
    uint32_t m_numConsElem = 0U;
    ElemAttr m_consElems[MAX_NUM_ELEMENTS];

    // Reconciled packet element atrribute
    uint32_t m_numElem = 0U;
    ElemAttr m_elems[MAX_NUM_ELEMENTS];

    // Packet element descriptions
    NvSciStreamPacket m_packetHandles[MAX_NUM_PACKETS];

    uint32_t m_numPacketReady = 0U;
    uint32_t m_numPackets = 0U;
    bool m_elementsDone;
    bool m_packetsDone;
    bool m_isC2C = false;
    std::vector<uint32_t> m_vElemTypesToSkip{};
    std::shared_ptr<CLateConsumerHelper> m_spLateConsHelper = nullptr;
};

#endif
