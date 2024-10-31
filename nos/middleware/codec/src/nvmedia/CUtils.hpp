/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "nvmedia_core.h"
#include "nvscierror.h"

#include <cstdarg>
#include <iostream>
#include <memory>
#include <vector>

#include "NvSIPLCamera.hpp"
#include "NvSIPLCommon.hpp"
#include "nvscibuf.h"

using namespace nvsipl;
using namespace std;

#ifndef CUTILS_HPP
#define CUTILS_HPP

typedef std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer*>> NvSIPLBuffers;

#define MAX_NUM_SURFACES (3U)

typedef struct {
    NvSciBufType bufType;
    uint64_t size;
    uint32_t planeCount;
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planeWidths[MAX_NUM_SURFACES];
    uint32_t planeHeights[MAX_NUM_SURFACES];
    uint32_t planePitches[MAX_NUM_SURFACES];
    uint32_t planeBitsPerPixels[MAX_NUM_SURFACES];
    uint32_t planeAlignedHeights[MAX_NUM_SURFACES];
    uint64_t planeAlignedSizes[MAX_NUM_SURFACES];
    uint8_t planeChannelCounts[MAX_NUM_SURFACES];
    uint64_t planeOffsets[MAX_NUM_SURFACES];
    uint64_t topPadding[MAX_NUM_SURFACES];
    uint64_t bottomPadding[MAX_NUM_SURFACES];
    bool needSwCacheCoherency;
    NvSciBufAttrValColorFmt planeColorFormats[MAX_NUM_SURFACES];
    uint32_t align[MAX_NUM_SURFACES];
} BufferAttrs;

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs& bufAttrs);

#endif
