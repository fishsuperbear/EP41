#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"

HWNvmediaOutputPipelineContext::HWNvmediaOutputPipelineContext(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
	CameraModuleInfo* i_pmoduleinfo, HWNvmediaSensorOutputPipelineProfiler* i_pprofiler)
{
	poutputpipeline_ops = i_poutputpipeline_ops;
	pmoduleinfo = i_pmoduleinfo;
	psensorinfo = &i_pmoduleinfo->sensorInfo;
	pprofiler = i_pprofiler;
	hw_plat_atomic_set_u64(&_framecount, 0);
	_peventhandler = new HWNvmediaEventHandlerOutputPipeline();
}

HWNvmediaOutputPipelineContext::~HWNvmediaOutputPipelineContext()
{
	delete(_peventhandler);
}

hw_ret_s32 HWNvmediaOutputPipelineContext::RegisterDataCallback_ToOutput(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
	HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig)
{
    HW_NVMEDIA_LOG_DEBUG("HWNvmediaOutputPipelineContext::RegisterDataCallback_ToOutput;\r\n");
	CConsumer* pconsumer;
	/*
	* Only support direct mode.
	*/
	if (!i_peventhandlercbconfig->bdirectcb)
	{
		HW_NVMEDIA_LOG_ERR("Only support direct data cb mode!\r\n");
		return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_DIRECTCB_MODE);
	}
	for (u32 i = 0U; i < _peventhandler->vector_pclients.size(); i++) {
		pconsumer = dynamic_cast<CConsumer*>(_peventhandler->vector_pclients[i].get());
		/*
		* @author shizifan
		* first client is producer, producer dynamic_cast to pconsumer will be failed
		*/
		if (pconsumer == nullptr)
		{
			continue;
		}
		if (pconsumer->GetConsumerType() == i_peventhandlercbconfig->consumertype || pconsumer->GetConsumerType()== VIC_CONSUMER)
		{
            HW_NVMEDIA_LOG_DEBUG("HWNvmediaOutputPipelineContext::RegisterDataCallback_ToOutput;consumerType=%d\r\n",pconsumer->GetConsumerType());
			CHK_LOG_SENTENCE_HW_RET_S32(pconsumer->RegisterDirectCb(i_pcbconfig, i_peventhandlercbconfig));
		}
	}
	return 0;
}

hw_ret_s32 HWNvmediaOutputPipelineContext::GetHandleFrame(hw_video_handleframe* o_phandleframe, u32 i_timeoutus, 
	HW_VIDEO_FRAMERETSTATUS* o_pframeretstatus)
{
	SIPLStatus status;
    size_t timeoutus = 200000;  //(size_t)((i_timeoutus == HW_TIMEOUT_FOREVER) ? HW_TIMEOUT_US_DEFAULT : i_timeoutus);
    size_t timeoutustotal = 0;
    NvSIPLBuffers& pBuffers = _siplbuffers;
    INvSIPLClient::INvSIPLBuffer *pbuf = nullptr;
	pBuffers.resize(_frameCompletionQueue.size());
    while (_peventhandler->brunning) {
        for (uint32_t index = 0U; index < _frameCompletionQueue.size(); ++index) {
            status = _frameCompletionQueue[index].second->Get(pbuf, timeoutus);

            if (status == NVSIPL_STATUS_OK) {
                pBuffers[index].first = _frameCompletionQueue[index].first;
                pBuffers[index].second = pbuf;
            } else if (status == NVSIPL_STATUS_TIMED_OUT) {
                if (i_timeoutus != HW_TIMEOUT_FOREVER)
                {
                    timeoutustotal += timeoutus;
                    if (timeoutustotal > i_timeoutus) {
                        HW_NVMEDIA_LOG_TRACE("Frame Queue timeout\r\n");
                        *o_pframeretstatus = HW_VIDEO_FRAMERETSTATUS_TIMEOUT;
                        return 0;
                    }
                }
                else
                {
                    HW_NVMEDIA_LOG_DEBUG("Frame Queue timeout\r\n");
                    break;
                }
                break;
            } else if (status == NVSIPL_STATUS_EOF) {
                HW_NVMEDIA_LOG_INFO("Frame Queue shutdown\r\n");
                _peventhandler->brunning = 0;
                *o_pframeretstatus = HW_VIDEO_FRAMERETSTATUS_QUIT;
                return 0;
            } else {
                HW_NVMEDIA_LOG_ERR("Unexpected frame queue return status\r\n");
                _peventhandler->brunning = 0;
                *o_pframeretstatus = HW_VIDEO_FRAMERETSTATUS_QUIT;
                return 0;
            }
        }
        if (status == NVSIPL_STATUS_OK) {
            *o_phandleframe = (hw_video_handleframe)&pBuffers;
            *o_pframeretstatus = HW_VIDEO_FRAMERETSTATUS_GET;
            return 0;
        }
	}
    *o_pframeretstatus = HW_VIDEO_FRAMERETSTATUS_QUIT;
    return 0;
}

hw_ret_s32 HWNvmediaOutputPipelineContext::Handle(hw_video_handleframe i_handleframe)
{
    SIPLStatus status;
    CProducer* pproducer = dynamic_cast<CProducer*>(_peventhandler->vector_pclients[0].get());
    CHK_PTR_AND_RET_S32(pproducer, "m_vClients[0] converts to CProducer");
    NvSIPLBuffers& pBuffers = (*(NvSIPLBuffers*)i_handleframe);
    status = pproducer->Post(&pBuffers);
    if (status != NVSIPL_STATUS_OK) {
        HW_NVMEDIA_LOG_UNMASK("HWNvmediaOutputPipelineContext::GetHandleFrame p->post status != NVSIPL_STATUS_OK\n");
        _peventhandler->brunning = 0;
        /*
		* We still need to do the release buffer operation. But do not check the return value of 
		* release operation.
		*/
        for (uint32_t i = 0; i < pBuffers.size(); ++i) {
            if (pBuffers[i].second != nullptr) {
                pBuffers[i].second->Release();
                pBuffers[i].second = nullptr;
            }
        }
        return HW_RET_S32_NVMEDIA_SIPLSTATUS(status);
    }
    for (uint32_t i = 0; i < pBuffers.size(); ++i) {
        if (pBuffers[i].second != nullptr) {
            status = pBuffers[i].second->Release();
            pBuffers[i].second = nullptr;
            if (status != NVSIPL_STATUS_OK) {
                _peventhandler->brunning = 0;
				return HW_RET_S32_NVMEDIA_SIPLSTATUS(status);
			}
        }
    }
    /*
	* Successfully handled.
	*/
    hw_plat_atomic_cas_exchangeadd_u64(&_framecount, 1, NULL, NULL);
    return 0;
}

hw_ret_s32 HWNvmediaOutputPipelineContext::SkipHandle(hw_video_handleframe i_handleframe)
{
    NvSIPLBuffers& pBuffers = (*(NvSIPLBuffers*)i_handleframe);
    for (uint32_t i = 0; i < pBuffers.size(); ++i) {
        if (pBuffers[i].second != nullptr) {
            pBuffers[i].second->Release();
            pBuffers[i].second = nullptr;
        }
    }
    return 0;
}

hw_ret_s32 HWNvmediaOutputPipelineContext::GetFrameTimeInfo(hw_video_handleframe i_handleframe, struct hw_video_buffertimeinfo_t* o_ptimeinfo) {
    NvSIPLBuffers& pBuffers = (*(NvSIPLBuffers*)i_handleframe);
    INvSIPLClient::INvSIPLBuffer* pbuffer = pBuffers[0].second;
    INvSIPLClient::INvSIPLNvMBuffer* pNvMBuf = reinterpret_cast<INvSIPLClient::INvSIPLNvMBuffer*>(pbuffer);
    const INvSIPLClient::ImageMetaData& md = pNvMBuf->GetImageData();
    o_ptimeinfo->framecapturetsc = md.frameCaptureTSC;
    return 0;
}

hw_ret_s32 HWNvmediaOutputPipelineContext::GetCount(u32* o_pcount)
{
	*o_pcount = (u32)_frameCompletionQueue[0].second->GetCount();
	return 0;
}

hw_ret_s32 HWNvmediaOutputPipelineContext::ResetFrameCounter()
{
	hw_plat_atomic_set_u64(&_framecount, 0);
	return 0;
}

hw_ret_s32 HWNvmediaOutputPipelineContext::GetFrameCount(u64* o_pframecount)
{
	hw_plat_atomic_get_u64(&_framecount, o_pframecount);
	//printf("framecount=%lld\r\n", *o_pframecount);
	//printf("profiler framecount=%lld\r\n", pprofiler->profdata.framecount);
	return 0;
}

static s32 hw_nvmedia_outputpipeline_gethandleframe(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
	hw_video_handleframe* o_phandleframe, u32 i_timeoutus, HW_VIDEO_FRAMERETSTATUS* o_pframeretstatus)
{
	HWNvmediaOutputPipelineContext* pnvmediaoutputpipeline = (HWNvmediaOutputPipelineContext*)i_poutputpipeline_ops->priv;
	return pnvmediaoutputpipeline->GetHandleFrame(o_phandleframe, i_timeoutus, o_pframeretstatus);
}

static s32 hw_nvmedia_outputpipeline_handle(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
	hw_video_handleframe i_handleframe)
{
	HWNvmediaOutputPipelineContext* pnvmediaoutputpipeline = (HWNvmediaOutputPipelineContext*)i_poutputpipeline_ops->priv;
	return pnvmediaoutputpipeline->Handle(i_handleframe);
}

static s32 hw_nvmedia_outputpipeline_skiphandle(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
	hw_video_handleframe i_handleframe)
{
	HWNvmediaOutputPipelineContext* pnvmediaoutputpipeline = (HWNvmediaOutputPipelineContext*)i_poutputpipeline_ops->priv;
	return pnvmediaoutputpipeline->SkipHandle(i_handleframe);
}

static s32 hw_nvmedia_outputpipeline_getframetimeinfo(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
    hw_video_handleframe i_handleframe, struct hw_video_buffertimeinfo_t* o_ptimeinfo) {
    HWNvmediaOutputPipelineContext* pnvmediaoutputpipeline = (HWNvmediaOutputPipelineContext*)i_poutputpipeline_ops->priv;
    return pnvmediaoutputpipeline->GetFrameTimeInfo(i_handleframe, o_ptimeinfo);
}

static s32 hw_nvmedia_outputpipeline_getcount(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops, u32* o_pcount)
{
	HWNvmediaOutputPipelineContext* pnvmediaoutputpipeline = (HWNvmediaOutputPipelineContext*)i_poutputpipeline_ops->priv;
	return pnvmediaoutputpipeline->GetCount(o_pcount);
}

static s32 hw_nvmedia_outputpipeline_resetframecounter(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops)
{
	HWNvmediaOutputPipelineContext* pnvmediaoutputpipeline = (HWNvmediaOutputPipelineContext*)i_poutputpipeline_ops->priv;
	return pnvmediaoutputpipeline->ResetFrameCounter();
}

static s32 hw_nvmedia_outputpipeline_getframecount(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops, u64* o_pframecount)
{
	HWNvmediaOutputPipelineContext* pnvmediaoutputpipeline = (HWNvmediaOutputPipelineContext*)i_poutputpipeline_ops->priv;
	return pnvmediaoutputpipeline->GetFrameCount(o_pframecount);
}

s32 hw_nvmedia_outputpipeline_setops(struct hw_video_outputpipeline_ops_t* io_poutputpipeline_ops)
{
	io_poutputpipeline_ops->gethandleframe = hw_nvmedia_outputpipeline_gethandleframe;
	io_poutputpipeline_ops->handle = hw_nvmedia_outputpipeline_handle;
	io_poutputpipeline_ops->skiphandle = hw_nvmedia_outputpipeline_skiphandle;
    io_poutputpipeline_ops->getframetimeinfo = hw_nvmedia_outputpipeline_getframetimeinfo;
	io_poutputpipeline_ops->getcount = hw_nvmedia_outputpipeline_getcount;
	io_poutputpipeline_ops->frame_ops.resetframecounter = hw_nvmedia_outputpipeline_resetframecounter;
	io_poutputpipeline_ops->frame_ops.getframecount = hw_nvmedia_outputpipeline_getframecount;
	return 0;
}
