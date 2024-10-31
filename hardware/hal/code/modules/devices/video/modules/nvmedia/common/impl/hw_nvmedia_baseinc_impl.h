#ifndef HW_NVMEDIA_BASEINC_IMPL_H
#define HW_NVMEDIA_BASEINC_IMPL_H

#include "hw_nvmedia_common.h"

#include <cstring>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <ctime>
#include <atomic>
#include <cmath>
#include <fstream>
#include <vector>
#include <memory>

#include "NvSIPLVersion.hpp" // Version
#include "NvSIPLTrace.hpp" // Trace
#include "NvSIPLQuery.hpp" // Query
#include "NvSIPLQueryTrace.hpp" // Query Trace
#include "NvSIPLCommon.hpp" // Common
#include "NvSIPLCamera.hpp" // Camera
#include "NvSIPLPipelineMgr.hpp" // Pipeline manager
#include "NvSIPLClient.hpp" // Client

using namespace nvsipl;

/*
* For multi isp use.
*/
typedef std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer *>> NvSIPLBuffers;

#endif
