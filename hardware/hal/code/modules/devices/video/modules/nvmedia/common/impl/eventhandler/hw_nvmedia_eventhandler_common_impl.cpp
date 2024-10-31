#include "hw_nvmedia_eventhandler_common_impl.h"

SIPLStatus GetConsumerTypeFromAppType(AppType appType, ConsumerType& consumerType)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    switch (appType) {
    case IPC_CUDA_CONSUMER:
        consumerType = CUDA_CONSUMER;
        break;
    case IPC_ENC_CONSUMER:
        consumerType = ENC_CONSUMER;
        break;
    default:
        status = NVSIPL_STATUS_BAD_ARGUMENT;
        break;
    }

    return status;
}

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs& bufAttrs)
{
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttrList;

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        { NvSciBufImageAttrKey_Size, NULL, 0 },               //0
        { NvSciBufImageAttrKey_Layout, NULL, 0 },             //1
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },         //2
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },         //3
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },        //4
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },         //5
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },  //6
        { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 }, //7
        { NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0 },   //8
        { NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0 },  //9
        { NvSciBufImageAttrKey_PlaneOffset, NULL, 0 },        //10
        { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },    //11
        { NvSciBufImageAttrKey_TopPadding, NULL, 0 },        //12
        { NvSciBufImageAttrKey_BottomPadding, NULL, 0 },   //13
        { NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0 } //14
    };

    err = NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjGetAttrList");
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, sizeof(imgAttrs) / sizeof(imgAttrs[0]));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListGetAttrs");

    bufAttrs.size = *(static_cast<const uint64_t*>(imgAttrs[0].value));
    bufAttrs.layout = *(static_cast<const NvSciBufAttrValImageLayoutType*>(imgAttrs[1].value));
    bufAttrs.planeCount = *(static_cast<const uint32_t*>(imgAttrs[2].value));
    bufAttrs.needSwCacheCoherency = *(static_cast<const bool*>(imgAttrs[14].value));

    memcpy(bufAttrs.planeWidths,
        static_cast<const uint32_t*>(imgAttrs[3].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeHeights,
        static_cast<const uint32_t*>(imgAttrs[4].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planePitches,
        static_cast<const uint32_t*>(imgAttrs[5].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeBitsPerPixels,
        static_cast<const uint32_t*>(imgAttrs[6].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedHeights,
        static_cast<const uint32_t*>(imgAttrs[7].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedSizes,
        static_cast<const uint64_t*>(imgAttrs[8].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeChannelCounts,
        static_cast<const uint8_t*>(imgAttrs[9].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeOffsets,
        static_cast<const uint64_t*>(imgAttrs[10].value),
        bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeColorFormats,
        static_cast<const NvSciBufAttrValColorFmt*>(imgAttrs[11].value),
        bufAttrs.planeCount * sizeof(NvSciBufAttrValColorFmt));
    memcpy(bufAttrs.topPadding,
        static_cast<const uint32_t*>(imgAttrs[12].value),
        bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.bottomPadding,
        static_cast<const uint32_t*>(imgAttrs[13].value),
        bufAttrs.planeCount * sizeof(uint32_t));

    //Print sciBuf attributes
    LOG_DBG("========PopulateBufAttr========\n");
    LOG_DBG("size=%lu, layout=%u, planeCount=%u\n",
        bufAttrs.size,
        bufAttrs.layout,
        bufAttrs.planeCount);
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        LOG_DBG("plane %u: planeWidth=%u, planeHeight=%u, planePitch=%u, planeBitsPerPixels=%u, planeAlignedHeight=%u\n",
            i,
            bufAttrs.planeWidths[i],
            bufAttrs.planeHeights[i],
            bufAttrs.planePitches[i],
            bufAttrs.planeBitsPerPixels[i],
            bufAttrs.planeAlignedHeights[i]);
        LOG_DBG("plane %u: planeAlignedSize=%lu, planeOffset=%lu, planeColorFormat=%u, planeChannelCount=%u\n",
            i,
            bufAttrs.planeAlignedSizes[i],
            bufAttrs.planeOffsets[i],
            bufAttrs.planeColorFormats[i],
            bufAttrs.planeChannelCounts[i]);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus  HandleException(std::exception* const e)
{
    if (nullptr == e)
    {
        return NVSIPL_STATUS_ERROR;
    }

    std::logic_error* logicErr = nullptr;
    std::runtime_error* runTimeErr = nullptr;
    std::bad_alloc* badAlloc = nullptr;

    logicErr = dynamic_cast<std::logic_error*>(e);
    if (nullptr != logicErr)
    {
        LOG_ERR("std::logic_error : %s\n", logicErr->what());
    }

    runTimeErr = dynamic_cast<std::runtime_error*>(e);
    if (nullptr != runTimeErr)
    {
        LOG_ERR("std::runtime_error : %s\n", runTimeErr->what());
    }

    badAlloc = dynamic_cast<std::bad_alloc*>(e);
    if (nullptr != badAlloc)
    {
        LOG_ERR("std::bad_alloc: %s\n", badAlloc->what());
    }
    LOG_ERR("std::exception : %s\n", e->what());

    return NVSIPL_STATUS_ERROR;
}