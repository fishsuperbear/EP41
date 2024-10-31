// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CFACTORY_HPP
#define CFACTORY_HPP

#include <cassert>
#include "hw_nvmedia_eventhandler_common_impl.h"

//#include "CUtils.hpp"
#include "CPoolManager.hpp"
#include "CSIPLProducer.hpp"
#include "CCudaConsumer.hpp"
#include "CEncConsumer.hpp"
#include "CCommonConsumer.hpp"
#include "CVICConsumer.hpp"
#include "CVICProducer.hpp"
#include "ICascadedProvider.hpp"
#include "CAttributeProvider.hpp"


#include "nvscibuf.h"

using namespace std;
using namespace nvsipl;

class CFactory
{
public:
    CFactory() {}
    ~CFactory();

    static std::shared_ptr<CAttributeProvider> CreateAttributeProvider(NvSciSyncModule syncModule, NvSciBufModule bufModule)
    {
        return std::shared_ptr<CAttributeProvider>(new CAttributeProvider(bufModule, syncModule));
    }

    static std::unique_ptr<CPoolManager> CreatePoolManager(uint32_t uSensor, uint32_t numPackets, std::shared_ptr<CAttributeProvider> attrProvider = nullptr);
    static std::vector<ElementInfo> GetElementsInfo(UseCaseInfo &ucInfo);
    static std::unique_ptr<CProducer> CreateVICProducer(NvSciStreamBlock poolHandle, uint32_t uSensor,
            ICascadedProvider* pCascadedProvider, UseCaseInfo &ucInfo);
    static std::unique_ptr<CProducer> CreateProducer(NvSciStreamBlock poolHandle, uint32_t uSensor,
            uint32_t i_outputtype, INvSIPLCamera* pCamera, UseCaseInfo &ucInfo, std::shared_ptr<CAttributeProvider> attrProvider = nullptr);
    static std::unique_ptr<CProducer> CreateIPCProducer(NvSciStreamBlock poolHandle, uint32_t uSensor,
            uint32_t i_outputtype, INvSIPLCamera* pCamera, UseCaseInfo &ucInfo, std::shared_ptr<CAttributeProvider> attrProvider = nullptr);
    static std::unique_ptr<CConsumer> CreateConsumer(ConsumerType consumerType, SensorInfo* pSensorInfo,
                                                     uint32_t i_outputtype, bool bUseMailbox, UseCaseInfo& ucInfo, HWNvmediaOutputPipelineContext* i_poutputpipeline, int encodeType = HW_VIDEO_REGDATACB_TYPE_AVC, void* i_pvicconsumer = nullptr);
    static SIPLStatus CreateMulticastBlock(uint32_t consumerCount, NvSciStreamBlock& multicastHandle);
    static SIPLStatus CreateIpcBlock(NvSciSyncModule syncModule, NvSciBufModule bufModule,
        const char* channel, bool isSrc, NvSciStreamBlock* ipcBlock, NvSciIpcEndpoint *ipcEndpoint = nullptr);
    static SIPLStatus ReleaseIpcBlock(NvSciStreamBlock ipcBlock, NvSciIpcEndpoint ipcEndpoint);
};

#endif

