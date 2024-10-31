// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CLATECONSUMERHELPER_H
#define CLATECONSUMERHELPER_H

#define WFD_NVX_create_source_from_nvscibuf
#define WFD_WFDEXT_PROTOTYPES

#include <WF/wfd.h>
#include <WF/wfdext.h>

#include "CUtils.hpp"

class CLateConsumerHelper
{
  public:
    CLateConsumerHelper() = delete;
    CLateConsumerHelper(NvSciBufModule &bufMod, NvSciSyncModule &syncMod)
        : m_bufModule(bufMod)
        , m_syncModule(syncMod)
    {
    }
    ~CLateConsumerHelper(){};

    SIPLStatus GetBufAttrLists(std::vector<NvSciBufAttrList> &outBufAttrList) const;
    SIPLStatus GetSyncWaiterAttrLists(std::vector<NvSciSyncAttrList> &outWaiterAttrList) const;
    uint32_t GetLateConsCount() const;

  private:
    static SIPLStatus GetCudaBufAttrList(NvSciBufAttrList bufAttrList);
    static SIPLStatus GetCudaSyncWaiterAttrList(NvSciSyncAttrList waiterAttrList);
    static SIPLStatus GetDisplayBufAttrList(NvSciBufAttrList bufAttrList);
    static SIPLStatus GetDisplaySyncWaiterAttrList(NvSciSyncAttrList waiterAttrList);

    NvSciBufModule m_bufModule = nullptr;
    NvSciSyncModule m_syncModule = nullptr;
};

#endif