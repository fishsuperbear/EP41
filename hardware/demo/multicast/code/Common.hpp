// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef COMMON_HPP
#define COMMON_HPP
    constexpr uint32_t MAX_NUM_SENSORS= 16U;
    constexpr uint32_t MAX_NUM_PACKETS = 6U;
    constexpr uint32_t MAX_NUM_ELEMENTS = 2U;
    constexpr uint32_t DATA_ELEMENT_INDEX = 0U;
    constexpr uint32_t META_ELEMENT_INDEX = 1U;
    constexpr uint32_t MAX_NUM_CONSUMERS = 3U;
    constexpr uint32_t MAX_WAIT_SYNCOBJ = MAX_NUM_CONSUMERS;
    constexpr uint32_t MAX_NUM_SYNCS = 8U;
    constexpr uint32_t MAX_QUERY_TIMEOUTS = 10U;
    constexpr int QUERY_TIMEOUT = 1000000; // usecs
    constexpr int QUERY_TIMEOUT_FOREVER = -1;
    constexpr uint32_t NVMEDIA_IMAGE_STATUS_TIMEOUT_MS = 100U;
    constexpr uint32_t DUMP_START_FRAME = 20U;
    constexpr uint32_t DUMP_END_FRAME = 39U;
    constexpr int64_t FENCE_FRAME_TIMEOUT_US = 100000U;
#endif
