// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef COPENWFD_CONTROLLER_HPP
#define COPENWFD_CONTROLLER_HPP

#define WFD_NVX_create_source_from_nvscibuf
#define WFD_WFDEXT_PROTOTYPES

#include <WF/wfd.h>
#include <WF/wfdext.h>

#include "nvscistream.h"
#include "CUtils.hpp"
#include "Common.hpp"

namespace hozon {
namespace netaos {
namespace desay {

class COpenWFDController
{
  public:
    COpenWFDController();
    ~COpenWFDController(void);

    const SIPLStatus InitResource(uint32_t numPorts = 1U);
    const SIPLStatus CreateWFDSource(NvSciBufObj &obj, uint32_t wfdPipelineIdx, uint32_t packetId);
    const SIPLStatus SetDisplayNvSciBufAttributesNVX(NvSciBufAttrList &attrList) const;
    const SIPLStatus SetDisplayNvSciSyncAttributesNVX(NvSciSyncAttrList &signalerAttrList,
                                                      NvSciSyncAttrList &waiterAttrList) const;
    const SIPLStatus InsertPrefence(uint32_t wfdPipelineIdx, uint32_t packetIndex, NvSciSyncFence &prefence) const;
    const SIPLStatus SetRect(uint32_t sourceX, uint32_t sourceY, uint32_t sourceWidth, uint32_t sourceHeight, 
        uint32_t dstWidth, uint32_t dstHeight, uint32_t wfdPipelineIdx) const;
    const SIPLStatus RegisterSignalSyncObj(NvSciSyncObj syncObj, uint32_t wfdPipelineIdx);
    const SIPLStatus RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) const;
    const SIPLStatus SetEofSyncObj(void) const;
    const SIPLStatus Flip(uint32_t wfdPipelineIdx, uint32_t packetId, NvSciSyncFence *pPostfence);
    const SIPLStatus DeInit(void);

  private:
    const SIPLStatus CreatePipeline(WFDPort port, uint32_t wfdPipelineIdx);

  private:
    WFDDevice m_wfdDevice = WFD_INVALID_HANDLE;
    WFDPort m_wfdPorts[MAX_NUM_WFD_PORTS] = { WFD_INVALID_HANDLE };
    WFDint m_wfdPortIds[MAX_NUM_WFD_PORTS] = { 0 };
    WFDPipeline m_wfdPipelines[MAX_NUM_WFD_PIPELINES] = { WFD_INVALID_HANDLE };
    WFDSource m_wfdSources[MAX_NUM_WFD_PIPELINES][MAX_NUM_PACKETS];
    WFDint m_windowWidths[MAX_NUM_WFD_PORTS] = { 0 };
    WFDint m_windowHeights[MAX_NUM_WFD_PORTS] = { 0 };
    WFDint m_wfdNumPorts = 0;
    uint32_t m_wfdNumPipelines = 0;
    std::string m_name;
};

}
}
}

#endif
