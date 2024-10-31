#ifndef HW_NVMEDIA_OUTPUTPIPELINE_IMPL_H
#define HW_NVMEDIA_OUTPUTPIPELINE_IMPL_H

#include "hw_nvmedia_blockpipeline_impl.h"
#include "CUtils.hpp"

class HWNvmediaEventHandlerOutputPipeline;

/*
* Define the priv of struct hw_video_outputpipeline_ops_t here.
*/
class HWNvmediaOutputPipelineContext
{
	friend class HWNvmediaContext;
	friend class HWNvmediaSingleProcessContext;
	friend class HWNvmediaIpcConsumerContext;
	friend class HWNvmediaIpcProducerContext;
private:
	HWNvmediaOutputPipelineContext();
public:
	HWNvmediaOutputPipelineContext(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
		CameraModuleInfo* i_pmoduleinfo, HWNvmediaSensorOutputPipelineProfiler* i_pprofiler);
	~HWNvmediaOutputPipelineContext();
public:
	// add tooutput description string to tell difference from sensor pipeline RegisterDataCallback func
	hw_ret_s32 RegisterDataCallback_ToOutput(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
		HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig);
public:
	hw_ret_s32 GetHandleFrame(hw_video_handleframe* o_phandleframe, u32 i_timeoutus, HW_VIDEO_FRAMERETSTATUS* o_pframeretstatus);
	hw_ret_s32 Handle(hw_video_handleframe i_handleframe);
	hw_ret_s32 SkipHandle(hw_video_handleframe i_handleframe);
    hw_ret_s32 GetFrameTimeInfo(hw_video_handleframe i_handleframe, struct hw_video_buffertimeinfo_t* o_ptimeinfo);
	hw_ret_s32 GetCount(u32* o_pcount);
	hw_ret_s32 GetLateConsCount(){ return _enableLateAttach ? NUM_IPC_LATE_CONS : 0; }
	void SetEnableLateAttach(bool o_enableLateAttach){ _enableLateAttach = o_enableLateAttach; }
	bool GetEnableLateAttach(){ return _enableLateAttach; }
    void SetLateattachIdxList(IdxManager* io_lateAttach_idxManager){ _lateAttach_idxManager.reset(io_lateAttach_idxManager); }
public:
	hw_ret_s32 ResetFrameCounter();
	hw_ret_s32 GetFrameCount(u64* o_pframecount);
public:
	struct hw_video_outputpipeline_ops_t*			poutputpipeline_ops;
	CameraModuleInfo*								pmoduleinfo;
	SensorInfo*										psensorinfo;
	HWNvmediaSensorOutputPipelineProfiler*			pprofiler;
    bool                                            pislateconsumerattached = false;
    std::unique_ptr<IdxManager>					    _lateAttach_idxManager;
    int                                             _server_fd;
    int                                             _client_fd;
private:
	INvSIPLFrameCompletionQueue*					_completequeue;
    /*
	* Due to multi isp.
	*/
    NvSIPLBuffers									_siplbuffers;
	std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLFrameCompletionQueue *>> _frameCompletionQueue;
protected:
	HWNvmediaEventHandlerOutputPipeline*			_peventhandler = nullptr;
private:
	hw_atomic_u64_t									_framecount;
    bool                                            _enableLateAttach = false;
};

s32 hw_nvmedia_outputpipeline_setops(struct hw_video_outputpipeline_ops_t* io_poutputpipeline_ops);

#endif
