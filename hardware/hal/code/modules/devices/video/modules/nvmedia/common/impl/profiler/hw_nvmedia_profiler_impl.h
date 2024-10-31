#ifndef HW_NVMEDIA_PROFILER_IMPL_H
#define HW_NVMEDIA_PROFILER_IMPL_H

#include "hw_nvmedia_outputpipeline_impl.h"

/*
* The instance of the class exist when device open.
*/
class HWNvmediaSensorOutputPipelineProfiler
{
public:
	typedef struct {
		hw_mutex_t							mutex_profdata;
		u64									framecount;
		u64									prevframecount;
	} ProfilingData;
public:
	HWNvmediaSensorOutputPipelineProfiler() = delete;
	HWNvmediaSensorOutputPipelineProfiler(u32 i_blockindex, u32 i_sensorindex, u32 i_sensorid, HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE i_outputtype)
	{
		blockindex = i_blockindex;
		sensorindex = i_sensorindex;
		sensorid = i_sensorid;
		outputtype = i_outputtype;

		hw_plat_mutex_init(&profdata.mutex_profdata, HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE);
		hw_plat_mutex_lock(&profdata.mutex_profdata);
		profdata.framecount = 0U;
		profdata.prevframecount = 0U;
		hw_plat_mutex_unlock(&profdata.mutex_profdata);
	}

	hw_ret_s32 OnFrameAvailable(void)
	{
		hw_plat_mutex_lock(&profdata.mutex_profdata);
		profdata.framecount++;
		hw_plat_mutex_unlock(&profdata.mutex_profdata);
		return 0;
	}

	~HWNvmediaSensorOutputPipelineProfiler()
	{
		hw_plat_mutex_deinit(&profdata.mutex_profdata);
	}

	u32										blockindex = UINT32_MAX;
	// sensor index of block
	u32										sensorindex = UINT32_MAX;
	// module.sensorInfo.id		module is vCameraModules in multicast sample
	u32										sensorid = UINT32_MAX;
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE		outputtype;
	ProfilingData							profdata;
};

#endif
