/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CUtils.hpp"
#include <cstring>

using namespace std;

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs& bufAttrs) {
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttrList;

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        {NvSciBufImageAttrKey_Size, NULL, 0},                       //0
        {NvSciBufImageAttrKey_Layout, NULL, 0},                     //1
        {NvSciBufImageAttrKey_PlaneCount, NULL, 0},                 //2
        {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},                 //3
        {NvSciBufImageAttrKey_PlaneHeight, NULL, 0},                //4
        {NvSciBufImageAttrKey_PlanePitch, NULL, 0},                 //5
        {NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0},          //6
        {NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0},         //7
        {NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0},           //8
        {NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0},          //9
        {NvSciBufImageAttrKey_PlaneOffset, NULL, 0},                //10
        {NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0},           //11
        {NvSciBufImageAttrKey_TopPadding, NULL, 0},                 //12
        {NvSciBufImageAttrKey_BottomPadding, NULL, 0},              //13
        {NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0},  //14
        {NvSciBufImageAttrKey_PlaneBaseAddrAlign, NULL, 0},         //15
        {NvSciBufImageAttrKey_LeftPadding, NULL, 0},                //16
        {NvSciBufImageAttrKey_RightPadding, NULL, 0},               //17
    };

    err = NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, sizeof(imgAttrs) / sizeof(imgAttrs[0]));

    bufAttrs.size = *(static_cast<const uint64_t*>(imgAttrs[0].value));
    bufAttrs.layout = *(static_cast<const NvSciBufAttrValImageLayoutType*>(imgAttrs[1].value));
    bufAttrs.planeCount = *(static_cast<const uint32_t*>(imgAttrs[2].value));
    bufAttrs.needSwCacheCoherency = *(static_cast<const bool*>(imgAttrs[14].value));

    memcpy(bufAttrs.planeWidths, static_cast<const uint32_t*>(imgAttrs[3].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeHeights, static_cast<const uint32_t*>(imgAttrs[4].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planePitches, static_cast<const uint32_t*>(imgAttrs[5].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeBitsPerPixels, static_cast<const uint32_t*>(imgAttrs[6].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedHeights, static_cast<const uint32_t*>(imgAttrs[7].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.planeAlignedSizes, static_cast<const uint64_t*>(imgAttrs[8].value), bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeChannelCounts, static_cast<const uint8_t*>(imgAttrs[9].value), bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeOffsets, static_cast<const uint64_t*>(imgAttrs[10].value), bufAttrs.planeCount * sizeof(uint64_t));
    memcpy(bufAttrs.planeColorFormats, static_cast<const NvSciBufAttrValColorFmt*>(imgAttrs[11].value), bufAttrs.planeCount * sizeof(NvSciBufAttrValColorFmt));
    memcpy(bufAttrs.topPadding, static_cast<const uint32_t*>(imgAttrs[12].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.bottomPadding, static_cast<const uint32_t*>(imgAttrs[13].value), bufAttrs.planeCount * sizeof(uint32_t));
    memcpy(bufAttrs.align, static_cast<const uint32_t*>(imgAttrs[15].value), bufAttrs.planeCount * sizeof(uint32_t));

    uint64_t padding[3] = {0};

    memcpy(padding, static_cast<const uint32_t*>(imgAttrs[12].value), bufAttrs.planeCount * sizeof(uint32_t));
    printf("top padding: ");
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        printf("%lu  ", padding[i]);
    }
    printf("\nbottom padding: ");
    memcpy(padding, static_cast<const uint32_t*>(imgAttrs[13].value), bufAttrs.planeCount * sizeof(uint32_t));
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        printf("%lu  ", padding[i]);
    }

    printf("\nleft padding: ");
    memcpy(padding, static_cast<const uint32_t*>(imgAttrs[16].value), bufAttrs.planeCount * sizeof(uint32_t));
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        printf("%lu  ", padding[i]);
    }

    printf("\nright padding: ");
    memcpy(padding, static_cast<const uint32_t*>(imgAttrs[17].value), bufAttrs.planeCount * sizeof(uint32_t));
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        printf("%lu  ", padding[i]);
    }
    printf("\n");

    //Print sciBuf attributes
    printf("========PopulateBufAttr========\n");
    printf("size=%lu, layout=%u, planeCount=%u\n", bufAttrs.size, bufAttrs.layout, bufAttrs.planeCount);
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        printf("plane %u: planeWidth=%u, planeHeight=%u, planePitch=%u, planeBitsPerPixels=%u, planeAlignedHeight=%u\n", i, bufAttrs.planeWidths[i], bufAttrs.planeHeights[i], bufAttrs.planePitches[i],
               bufAttrs.planeBitsPerPixels[i], bufAttrs.planeAlignedHeights[i]);
        printf("plane %u: align %u,  planeAlignedSize=%lu, planeOffset=%lu, planeColorFormat=%u, planeChannelCount=%u\n", i, bufAttrs.align[i], bufAttrs.planeAlignedSizes[i], bufAttrs.planeOffsets[i],
               bufAttrs.planeColorFormats[i], bufAttrs.planeChannelCounts[i]);
    }

    return NVSIPL_STATUS_OK;
}
