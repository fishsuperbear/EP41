// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CATTRIBUTEPROVIDER_H
#define CATTRIBUTEPROVIDER_H

#include <cassert>

#include <hw_nvmedia_eventhandler_common_impl.h>

/* #include "CUtils.hpp" */
#include "CCudaConsumer.hpp"

class CAttributeProvider
{
public:
    CAttributeProvider() = delete;
    CAttributeProvider(NvSciBufModule& bufMod, NvSciSyncModule& syncMod) : m_bufModule(bufMod), m_syncModule(syncMod)
    {
    }
    ~CAttributeProvider() {};

    SIPLStatus GetBufAttrList(ConsumerType type, NvSciBufAttrList *outBufAttrList)
    {
        assert(outBufAttrList != nullptr);

        NvSciBufAttrList bufAttrList = nullptr;
        NvSciError sciErr = NvSciBufAttrListCreate(m_bufModule, &bufAttrList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");

        SIPLStatus status = NVSIPL_STATUS_OK;
        switch (type) {
        case ConsumerType::CUDA_CONSUMER:
            status = CCudaConsumer::GetBufAttrList(bufAttrList);
            CHK_STATUS_AND_RETURN(status, "CCudaConsumer::GetBufAttrList");
            break;
        case ConsumerType::ENC_CONSUMER:
            // Not support by present
            assert(false);
            break;
        default:
            break;
        }
        *outBufAttrList = bufAttrList;
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus GetSyncWaiterAttrList(ConsumerType type, NvSciSyncAttrList* outWaiterAttrList)
    {
        assert(outWaiterAttrList != nullptr);

        NvSciSyncAttrList syncAttrList = nullptr;
        auto sciErr = NvSciSyncAttrListCreate(m_syncModule, &syncAttrList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler NvSciSyncAttrListCreate");

        SIPLStatus status = NVSIPL_STATUS_OK;
        switch (type) {
        case ConsumerType::CUDA_CONSUMER:
            status = CCudaConsumer::GetSyncWaiterAttrList(syncAttrList);
            CHK_STATUS_AND_RETURN(status, "CCudaConsumer::GetSyncWaiterAttrList");
            break;
        case ConsumerType::ENC_CONSUMER:
            // Not support by present
            assert(false);
            break;
        default:
            break;
        }
        *outWaiterAttrList = syncAttrList;
        return NVSIPL_STATUS_OK;
    }

private:
    NvSciBufModule m_bufModule = nullptr;
    NvSciSyncModule m_syncModule = nullptr;
};

#endif
