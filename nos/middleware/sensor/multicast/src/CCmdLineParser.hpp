// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <iomanip>

#include "CUtils.hpp"
#include "NvSIPLTrace.hpp" // NvSIPLTrace to set library trace level
#if !NV_IS_SAFETY
#include "NvSIPLQuery.hpp" // NvSIPLQuery to display platform config
#endif
#include "CAppConfig.hpp"
#include "platform/ar0820.hpp"

#ifndef CCMDLINEPARSER_HPP
#define CCMDLINEPARSER_HPP

using namespace std;
using namespace nvsipl;

class CCmdLineParser
{
  public:
    SIPLStatus Parse(int argc, char *argv[], CAppConfig *pAppConfig);

  private:
    void ShowUsage(void);
    void ShowConfigs(void);
    SIPLStatus ParseConsumerTypeStr(string sConsumerType, ConsumerType &eConsumerType);
    SIPLStatus ParseQueueTypeStr(string sQueueType, QueueType &eQueueType);
};

#endif
