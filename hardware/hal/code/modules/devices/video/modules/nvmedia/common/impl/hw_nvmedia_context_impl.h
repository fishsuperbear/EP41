#ifndef HW_NVMEDIA_CONTEXT_IMPL_H
#define HW_NVMEDIA_CONTEXT_IMPL_H

#include "hw_nvmedia_log_impl.h"
#include "hw_nvmedia_eventhandler_common_impl.h"

class HWNvmediaBlockPipelineContext;
class HWNvmediaSensorOutputPipelineProfiler;
class HWNvmediaOutputPipelineContext;

class CEventHandler;
class HWNvmediaEventHandlerRegDataCbConfig;

#define HW_NVMEDIA_NUMOUTPUTPIPELINES_MAX					64
constexpr uint32_t NUM_IPC_LATE_CONS = 5U;
STATIC_ASSERT(HW_NVMEDIA_NUMOUTPUTPIPELINES_MAX == HW_VIDEO_NUMBLOCKS_MAX * HW_VIDEO_NUMSENSORS_PER_BLOCK * HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_COUNT);

/*
* Define the priv of struct hw_video_t here.
* One device is correspondent to one instance here.
*/
class HWNvmediaContext
{
	friend class HWNvmediaBlockPipelineContext;
	friend class HWNvmediaSensorPipelineContext;
private:
	HWNvmediaContext();
public:
	HWNvmediaContext(struct hw_video_t* i_pvideo)
	{
		_pvideo = i_pvideo;
		hw_plat_atomic_set_u32(&_busingdevice, 0);
		hw_plat_atomic_set_u32(&_busingpipeline, 0);
		hw_plat_atomic_set_u32(&_bcancallpipelinestart, 0);
	}
	virtual ~HWNvmediaContext()
	{

	}
public:
	// called immediately after constructor. When return value is not 0, you can not continue do the operation.
	virtual hw_ret_s32 Init();
public:
	/*
	* You can set o_pinfo to NULL to mean you do not care about the info.
	* We make certain the mode like SINGLE_PROCESS or IPC when call the function.
	*/
	virtual hw_ret_s32 Device_Open(struct hw_video_info_t* o_pinfo) = 0;
	virtual hw_ret_s32 Device_Close() = 0;
	// i_pblockspipelineconfig may be NULL to mean default config
	virtual hw_ret_s32 Pipeline_Open(struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig,
		struct hw_video_blockspipeline_ops_t** o_ppblockspipeline_ops) = 0;
	virtual hw_ret_s32 Pipeline_Close() = 0;
	virtual hw_ret_s32 Pipeline_Start() = 0;
	virtual hw_ret_s32 Pipeline_Stop() = 0;
public:
	// used by deviceopen only, see MasterSetup in nvmedia multicast sample
	virtual hw_ret_s32 ModuleOpen() = 0;
	virtual hw_ret_s32 ModuleClose() = 0;
	// used by pipelineprepare only, create blocks in the output pipeline context
	virtual hw_ret_s32 CreateBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline) = 0;
	// used by pipelineclose only, according to nvidia multicast sample
	virtual hw_ret_s32 DestroyBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline) = 0;
	// used by pipelineprepare only
	virtual hw_ret_s32 ConnectBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline) = 0;
	// used by pipelineprepare only
	virtual hw_ret_s32 InitBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline) = 0;
	// used by pipelineprepare only, the implement is common, so implement it in HWNvmediaContext
	virtual hw_ret_s32 Reconcile(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by Reconcile only
	virtual hw_ret_s32 GetEventHandlerVector_Reconcile(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers) = 0;
	virtual hw_ret_s32 GetEventHandlerVector_Start(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers) = 0;
public:
	// used by HWNvmediaSensorPipelineContext RegisterDataCallback only, module so specifically implemented
	virtual hw_ret_s32 GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype,
		HW_VIDEO_INPUT_TYPE inputType, HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig) = 0;
private:
	static void EventHandlerThreadFunc(CEventHandler* i_peventhandler, HWNvmediaOutputPipelineContext* io_poutputpipeline);
public:
	virtual hw_ret_s32 StartEventHandler(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	virtual hw_ret_s32 StopEventHandler(HWNvmediaOutputPipelineContext* io_poutputpipeline);
public:
	// used by pipelineprepare only, we implement the default nito operation here in HWNvmediaContext
	virtual hw_ret_s32 NITOSetup(HWNvmediaOutputPipelineContext* io_poutputpipeline);
private:
	SIPLStatus loadnitofile(std::string i_nitofolderpath, std::string moduleName, std::vector<uint8_t>& nito);
private:
	/*
	* @return: 0 means success,
	* See hw_ret_s32 for details.
	*/
	hw_ret_s32 queryprepare();
	hw_ret_s32 checkversion();
private:
	// used by updateplatformcfg_perboardmodel only
	SIPLStatus checksku(const std::string& findStr, bool& bFound);
	// used by function deviceopen only
	hw_ret_s32 updateplatformcfg_perboardmodel();
protected:
	hw_ret_s32 deviceopen();
	hw_ret_s32 videoinfoget(struct hw_video_info_t* o_pinfo);
	hw_ret_s32 deviceclose();
	// do not prepare the event handler in the function
	hw_ret_s32 pipelineopen(struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig);
	hw_ret_s32 pipelineclose();
	hw_ret_s32 pipelineprepare();
	hw_ret_s32 pipelineunprepare();
	hw_ret_s32 pipelinestart();
	hw_ret_s32 pipelinestop();
private:
	// used by pipelineopen only
	hw_ret_s32 outputpipelineadd(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops);
protected:
	struct hw_video_t*						_pvideo;
	HWNvmediaDeviceOpenPara					_deviceopenpara;
	std::unique_ptr<INvSIPLQuery>			_pquery;
	PlatformCfg								_platformcfg;
	u32										_numblocks = 0U;
	struct hw_video_blockinfo_t				_parrayblockinfo[HW_VIDEO_NUMBLOCKS_MAX];
	// count of camera modules
	u32										_numsensors = 0U;
	// one module correspondent to one sensor
	std::vector<CameraModuleInfo>			_vcameramodules;
	// currently always NULL, reserved for future
	void*									_pvideoinfoext;
	std::unique_ptr<INvSIPLCamera>			_pcamera;
	NvSciBufModule							_scibufmodule = nullptr;
	NvSciSyncModule							_scisyncmodule = nullptr;
	NvSIPLDeviceBlockQueues					_deviceblockqueues;
	// when input config pointer is not null, it is the same as input config when call pipeline open
	struct hw_video_blockspipelineconfig_t	_blockspipelineconfig;
	struct hw_video_blockspipeline_ops_t*	_pblockspipeline_ops = nullptr;
protected:
	// we often use it to iterate doing operation
	struct hw_video_outputpipeline_ops_t*	_parraypoutputpipeline[HW_NVMEDIA_NUMOUTPUTPIPELINES_MAX] { nullptr };
	// the array valid pointer number of _parraypoutputpipeline
	u32										_numoutputpipeline = 0U;
protected:
	/*
	* 0 or 1
	* Only one thread can call deviceopen simultaneously.
	* Set to 1 when deviceopen, set back to 0 when deviceclose.
	*/
	hw_atomic_u32_t							_busingdevice;
	/*
	* 0 or 1
	* Only one thread can use pipeline simultaneously.
	* Set to 1 when pipelineopen, set back to 0 when pipelineclose.
	*/
	hw_atomic_u32_t							_busingpipeline;
	/*
	* 0 or 1, tag the status that cannot call pipelinestart.
	* Set to 1 only when pipelineprepare success, after called or other case it is 0.
	*/
	hw_atomic_u32_t							_bcancallpipelinestart;
protected:
	// profiler use, exist when device open, not valid when device close
	HWNvmediaSensorOutputPipelineProfiler*	_ppparraypoutputpipelineprofiler[HW_VIDEO_NUMBLOCKS_MAX][HW_VIDEO_NUMSENSORS_PER_BLOCK][HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_COUNT] { nullptr };
private:
	// 0 or 1, when pipelineprepare reset to 1
	u32										_bfirstblob = 1U;
	// for performance enhance use
	std::string								_lastmodulename_blob;
	// for performance enhance use
	std::vector<uint8_t>					_blob;
};

class HWNvmediaSingleProcessContext : public HWNvmediaContext
{
private:
	HWNvmediaSingleProcessContext();
public:
	HWNvmediaSingleProcessContext(struct hw_video_t* i_pvideo)
		: HWNvmediaContext(i_pvideo)
	{

	}
	virtual ~HWNvmediaSingleProcessContext()
	{

	}
public:
	hw_ret_s32 Device_Open(struct hw_video_info_t* o_pinfo);
	hw_ret_s32 Device_Close();
	hw_ret_s32 Pipeline_Open(struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig,
		struct hw_video_blockspipeline_ops_t** o_ppblockspipeline_ops);
	hw_ret_s32 Pipeline_Close();
	hw_ret_s32 Pipeline_Start();
	hw_ret_s32 Pipeline_Stop();
public:
	// used by deviceopen only, see MasterSetup in nvmedia multicast sample
	virtual hw_ret_s32 ModuleOpen();
	virtual hw_ret_s32 ModuleClose();
	// used by pipelineprepare only, create blocks in the output pipeline context
	virtual hw_ret_s32 CreateBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineclose only, according to nvidia multicast sample
	virtual hw_ret_s32 DestroyBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineprepare only
	virtual hw_ret_s32 ConnectBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineprepare only
	virtual hw_ret_s32 InitBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by Reconcile only
	virtual hw_ret_s32 GetEventHandlerVector_Reconcile(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers);
	virtual hw_ret_s32 GetEventHandlerVector_Start(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers);
public:
	// used by HWNvmediaSensorPipelineContext RegisterDataCallback only, module so specifically implemented
	virtual hw_ret_s32 GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype,
		HW_VIDEO_INPUT_TYPE inputType, HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig) = 0;
};

class HWNvmediaIpcConsumerContext : public HWNvmediaContext
{
private:
	HWNvmediaIpcConsumerContext();
public:
	HWNvmediaIpcConsumerContext(struct hw_video_t* i_pvideo)
		: HWNvmediaContext(i_pvideo)
	{

	}
	virtual ~HWNvmediaIpcConsumerContext()
	{

	}
public:
	hw_ret_s32 Device_Open(struct hw_video_info_t* o_pinfo);
	hw_ret_s32 Device_Close();
	hw_ret_s32 Pipeline_Open(struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig,
		struct hw_video_blockspipeline_ops_t** o_ppblockspipeline_ops);
	hw_ret_s32 Pipeline_Close();
	hw_ret_s32 Pipeline_Start();
	hw_ret_s32 Pipeline_Stop();
public:
	// used by deviceopen only, see MasterSetup in nvmedia multicast sample
	virtual hw_ret_s32 ModuleOpen();
	virtual hw_ret_s32 ModuleClose();
	// used by pipelineprepare only, create blocks in the output pipeline context
	virtual hw_ret_s32 CreateBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineclose only, according to nvidia multicast sample
	virtual hw_ret_s32 DestroyBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineprepare only
	virtual hw_ret_s32 ConnectBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineprepare only
	virtual hw_ret_s32 InitBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by Reconcile only
	virtual hw_ret_s32 GetEventHandlerVector_Reconcile(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers);
	virtual hw_ret_s32 GetEventHandlerVector_Start(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers);
public:
	// used by HWNvmediaSensorPipelineContext RegisterDataCallback only, module so specifically implemented
	virtual hw_ret_s32 GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype,
		HW_VIDEO_INPUT_TYPE inputType, HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig) = 0;

private:
	hw_ret_s32 CheckNvSciConnectAndReturn(NvSciError sciErr, NvSciStreamEventType event, std::string api, std::string name);
    s32 _cuda_idx;
};

class HWNvmediaIpcProducerContext : public HWNvmediaContext
{
private:
	HWNvmediaIpcProducerContext();
public:
	HWNvmediaIpcProducerContext(struct hw_video_t* i_pvideo)
		: HWNvmediaContext(i_pvideo)
	{

	}
	virtual ~HWNvmediaIpcProducerContext()
	{

	}
public:
	hw_ret_s32 Device_Open(struct hw_video_info_t* o_pinfo);
	hw_ret_s32 Device_Close();
	hw_ret_s32 Pipeline_Open(struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig,
		struct hw_video_blockspipeline_ops_t** o_ppblockspipeline_ops);
	hw_ret_s32 Pipeline_Close();
	hw_ret_s32 Pipeline_Start();
	hw_ret_s32 Pipeline_Stop();
public:
	// used by deviceopen only, see MasterSetup in nvmedia multicast sample
	virtual hw_ret_s32 ModuleOpen();
	virtual hw_ret_s32 ModuleClose();
	// used by pipelineprepare only, create blocks in the output pipeline context
	virtual hw_ret_s32 CreateBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineclose only, according to nvidia multicast sample
	virtual hw_ret_s32 DestroyBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineprepare only
	virtual hw_ret_s32 ConnectBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by pipelineprepare only
	virtual hw_ret_s32 InitBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline);
	// used by Reconcile only
	virtual hw_ret_s32 GetEventHandlerVector_Reconcile(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers);
	virtual hw_ret_s32 GetEventHandlerVector_Start(HWNvmediaOutputPipelineContext* io_poutputpipeline,
		std::vector<CEventHandler*>& i_vector_eventhandlers);
public:
	// used by HWNvmediaSensorPipelineContext RegisterDataCallback only, module so specifically implemented
	virtual hw_ret_s32 GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype,
		HW_VIDEO_INPUT_TYPE inputType, HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig) = 0;
    virtual void SetHwEvent(struct hw_event_t* io_pevent){m_pevent = io_pevent;};

private:
	hw_ret_s32 CheckNvSciConnectAndReturn(NvSciError sciErr, NvSciStreamEventType event, std::string api, std::string name);
    void Attach(HWNvmediaOutputPipelineContext *io_poutputpipeline,int channel_id);
    void Detach(HWNvmediaOutputPipelineContext *io_poutputpipeline,int channel_id);
    static void AttachEventListen(HWNvmediaIpcProducerContext* context,HWNvmediaOutputPipelineContext *io_poutputpipeline);
    static void ProcessClientEventHandle(HWNvmediaIpcProducerContext* context,HWNvmediaOutputPipelineContext *io_poutputpipeline,int clientfd);

private:
    NvSciIpcEndpoint m_lateIpcEndpoint = 0U;
    struct hw_event_t* m_pevent = nullptr;
};

#endif
