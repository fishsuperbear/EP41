#ifndef HW_NVMEDIA_EVENTHANDLER_OUTPUTPIPELINE_IMPL_H
#define HW_NVMEDIA_EVENTHANDLER_OUTPUTPIPELINE_IMPL_H

#include "hw_nvmedia_eventhandler_nvidia_impl.h"

class HWNvmediaEventHandlerOutputPipeline
{
public:
	// do the block delete operation according to nvidia multicast sample code
	~HWNvmediaEventHandlerOutputPipeline();
public:
	unique_ptr<CPoolManager>						ppoolmanager = nullptr;
	vector<unique_ptr<CClientCommon>>				vector_pclients;
	NvSciStreamBlock								block_multicast = 0U;
    NvSciStreamBlock                                pdstIpcHandle = 0;
    NvSciStreamBlock                                psrcIpcHandles[NUM_IPC_CONSUMERS];
    NvSciIpcEndpoint                                plateIpcEndpoint[NUM_IPC_CONSUMERS];
public:
	// 0 or 1
	u32												brunning = 0U;
	// event handlers thread
	vector<unique_ptr<thread>>						vector_threads;
};

class HWNvmediaEventHandlerRegDataCbCommonConsumerConfig
{
public:
	HW_VIDEO_BUFFERFORMAT_SUBTYPE					expectedoriginsubtype;
	// 0 or 1, always 0 when mode except yuv420bl origin sub type
	u32												bneedgetpixel;
};

/*
* Describe how to config the path and data callback function.
*/
class HWNvmediaEventHandlerRegDataCbConfig
{
public:
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE				outputtype;
	ConsumerType									consumertype;
	/*
	* 1 mean the data cb way is to do nothing but to call the cb directly. And only one to
	* one.
	* 0 mean complex mode, currently not support
	*/
	u32												bdirectcb;
	/*
	* Valid when consumertype is COMMON_CONSUMER.
	*/
	HWNvmediaEventHandlerRegDataCbCommonConsumerConfig	commonconsumerconfig;
};

#endif
