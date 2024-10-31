// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "Common.hpp"
#include "CLateConsumerHelper.hpp"
// #include "CCudaConsumer.hpp"
#include "CUtils.hpp"

SIPLStatus CLateConsumerHelper::GetBufAttrLists(std::vector<NvSciBufAttrList> &outBufAttrList) const
{
    // cuda consumer attached lately
    NvSciBufAttrList bufAttrList = nullptr;
    NvSciError sciErr = NvSciBufAttrListCreate(m_bufModule, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");

    SIPLStatus status = GetCudaBufAttrList(bufAttrList);
    if (status != NVSIPL_STATUS_OK) {
        NvSciBufAttrListFree(bufAttrList);
        LOG_ERR("GetCudaBufAttrList failed, status: %u\n", (status));
        return (status);
    }
    outBufAttrList.push_back(bufAttrList);

    // display consumer attached lately
    NvSciBufAttrList bufAttrListDisplay = nullptr;
    sciErr = NvSciBufAttrListCreate(m_bufModule, &bufAttrListDisplay);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");

    status = GetDisplayBufAttrList(bufAttrListDisplay);
    if (status != NVSIPL_STATUS_OK) {
        NvSciBufAttrListFree(bufAttrListDisplay);
        LOG_ERR("GetDisplayBufAttrList failed, status: %u\n", (status));
        return (status);
    }
    outBufAttrList.push_back(bufAttrListDisplay);
    return NVSIPL_STATUS_OK;
}

SIPLStatus CLateConsumerHelper::GetSyncWaiterAttrLists(std::vector<NvSciSyncAttrList> &outWaiterAttrList) const
{
    // cuda consumer attached lately
    NvSciSyncAttrList syncAttrList = nullptr;
    auto sciErr = NvSciSyncAttrListCreate(m_syncModule, &syncAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler NvSciSyncAttrListCreate");

    SIPLStatus status = GetCudaSyncWaiterAttrList(syncAttrList);
    if (status != NVSIPL_STATUS_OK) {
        NvSciSyncAttrListFree(syncAttrList);
        LOG_ERR("GetCudaSyncWaiterAttrList failed, status: %u\n", (status));
        return (status);
    };
    outWaiterAttrList.push_back(syncAttrList);

    // display consumer attached lately
    NvSciSyncAttrList syncAttrListDisplay = nullptr;
    sciErr = NvSciSyncAttrListCreate(m_syncModule, &syncAttrListDisplay);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler NvSciSyncAttrListCreate");

    status = GetDisplaySyncWaiterAttrList(syncAttrListDisplay);
    if (status != NVSIPL_STATUS_OK) {
        NvSciSyncAttrListFree(syncAttrListDisplay);
        LOG_ERR("GetDisplaySyncWaiterAttrList failed, status: %u\n", (status));
        return (status);
    };
    outWaiterAttrList.push_back(syncAttrListDisplay);
    return NVSIPL_STATUS_OK;
}

uint32_t CLateConsumerHelper::GetLateConsCount() const
{
    return NUM_IPC_CONSUMERS - 1;
}

SIPLStatus CLateConsumerHelper::GetCudaBufAttrList(NvSciBufAttrList bufAttrList)
{
    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

    cudaStatus = cudaSetDevice(0);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");

    NvSciRmGpuId gpuId;
    CUuuid uuid;
    auto cudaErr = cuDeviceGetUuid(&uuid, 0);
    CHK_CUDAERR_AND_RETURN(cudaErr, "cuDeviceGetUuid");
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    bool cpuaccess_flag = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
    };

    NvSciError sciErr =
        NvSciBufAttrListSetAttrs(bufAttrList, bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    return NVSIPL_STATUS_OK;
}

SIPLStatus CLateConsumerHelper::GetCudaSyncWaiterAttrList(NvSciSyncAttrList waiterAttrList)
{
    if (!waiterAttrList) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    auto cudaStatus = cudaDeviceGetNvSciSyncAttributes(waiterAttrList, 0, cudaNvSciSyncAttrWait);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CLateConsumerHelper::GetDisplayBufAttrList(NvSciBufAttrList bufAttrList) {
    // Default buffer attributes
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_Readonly;

    bool needCpuAccessFlag = false;
    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &bufPerm, sizeof(bufPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccessFlag, sizeof(needCpuAccessFlag) },
        { NvSciBufImageAttrKey_ScanType, &bufScanType, sizeof(bufScanType) },
    };

    WFDErrorCode wfdErr = wfdNvSciBufSetDisplayAttributesNVX(&bufAttrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciBufSetDisplayAttributesNVX");

    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CLateConsumerHelper::GetDisplaySyncWaiterAttrList(NvSciSyncAttrList waiterAttrList) {
    WFDErrorCode wfdErr = wfdNvSciSyncSetWaiterAttributesNVX(&waiterAttrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciSyncSetWaiterAttributesNVX");

    NvSciSyncAccessPerm accessPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair syncWaiterAttrs[] = {
        { NvSciSyncAttrKey_RequiredPerm, (void *)&accessPerm, sizeof(accessPerm) },
    };

    auto sciErr = NvSciSyncAttrListSetAttrs(waiterAttrList, syncWaiterAttrs,
                                            sizeof(syncWaiterAttrs) / sizeof(NvSciSyncAttrKeyValuePair));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");
    return NVSIPL_STATUS_OK;
}
