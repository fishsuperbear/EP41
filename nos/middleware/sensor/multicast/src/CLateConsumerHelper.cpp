// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CLateConsumerHelper.hpp"
#include "CCudaConsumer.hpp"
#include "CUtils.hpp"

SIPLStatus CLateConsumerHelper::GetBufAttrLists(std::vector<NvSciBufAttrList> &outBufAttrList) const
{
    NvSciBufAttrList bufAttrList = nullptr;
    NvSciError sciErr = NvSciBufAttrListCreate(m_bufModule, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");

    // Now, we only support cuda consumer attached lately
    SIPLStatus status = CCudaConsumer::GetBufAttrList(bufAttrList);
    if (status != NVSIPL_STATUS_OK) {
        NvSciBufAttrListFree(bufAttrList);
        LOG_ERR("CCudaConsumer::GetBufAttrList failed, status: %u\n", (status));
        return (status);
    }
    outBufAttrList.push_back(bufAttrList);
    return NVSIPL_STATUS_OK;
}

SIPLStatus CLateConsumerHelper::GetSyncWaiterAttrLists(std::vector<NvSciSyncAttrList> &outWaiterAttrList) const
{
    NvSciSyncAttrList syncAttrList = nullptr;
    auto sciErr = NvSciSyncAttrListCreate(m_syncModule, &syncAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler NvSciSyncAttrListCreate");

    SIPLStatus status = CCudaConsumer::GetSyncWaiterAttrList(syncAttrList);
    if (status != NVSIPL_STATUS_OK) {
        NvSciSyncAttrListFree(syncAttrList);
        LOG_ERR("CCudaConsumer::GetSyncWaiterAttrList failed, status: %u\n", (status));
        return (status);
    };
    outWaiterAttrList.push_back(syncAttrList);
    return NVSIPL_STATUS_OK;
}

uint32_t CLateConsumerHelper::GetLateConsCount() const
{
    return m_pAppConfig->IsLateAttachEnabled() ? 1 : 0;
}