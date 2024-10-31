#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"

hw_ret_s32 HWNvmediaSensorPipelineContext::Init()
{
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::RegisterDataCallback()
{
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaSensorPipelineContext::RegisterDataCallback;\r\n");
    /*
    * One regdatacb type should at most be registered once.
    */
    u32 bregistered[HW_VIDEO_REGDATACB_TYPE_COUNT] = { 0 };
    /*
    * Currently one outputtype should be at most be registered once.
    */
    u32 boutputregisterd[HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_COUNT] = { 0 };
    /*
    * Check the input sensor pipeline datacb config and register data cb function.
    */
    struct hw_video_sensorpipelineconfig_t* psensor =
        &pcontext->_blockspipelineconfig.parrayblock[psensorpipeline_ops->blockindex].parraysensor[psensorpipeline_ops->sensorindex];
    struct hw_video_sensorpipelinedatacbsconfig_t* pconfig = &psensor->datacbsconfig;
    u32 configi;
    struct hw_video_sensorpipelinedatacbconfig_t* pdatacbconfig;
    HWNvmediaEventHandlerRegDataCbConfig cbconfig;
    HWNvmediaOutputPipelineContext* poutputpipelinecontext;
    HW_NVMEDIA_LOG_DEBUG("HWNvmediaSensorPipelineContext::RegisterDataCallback;cb num=%d\r\n",pconfig->arraynumdatacbs);
    for (configi = 0; configi < pconfig->arraynumdatacbs; configi++)
    {
        pdatacbconfig = &pconfig->parraydatacbs[configi];
        if (pdatacbconfig->bused == 0) {
            continue;
        }
        if (bregistered[pdatacbconfig->type] != 0) {
            HW_NVMEDIA_LOG_ERR("The regdatacbtype[%u] registers twice!\r\n", pdatacbconfig->type);
            return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_REGISTER_TWICE);
        }
        CHK_LOG_SENTENCE_HW_RET_S32(pcontext->GetRegDataCbType_OutputType((HW_VIDEO_REGDATACB_TYPE)pdatacbconfig->type, 
            psensorpipeline_ops->inputType, &cbconfig));
        if (boutputregisterd[cbconfig.outputtype] != 0) {
            HW_NVMEDIA_LOG_ERR("The outputtype[%u] registers twice!\r\n", cbconfig.outputtype);
            return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_REGISTER_TWICE);
        }
        /*
        * Need to check whether the outputtype has instance.
        */
        switch (cbconfig.outputtype)
        {
        case HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP:
            if (psensor->bcaptureoutputrequested != 1) {
                HW_NVMEDIA_LOG_ERR("The config of icp not enable, but register icp datacb!\r\n");
                return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_NOT_ENABLE_BUT_REGISTER_DATACB);
            }
            break;
        case HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0:
            if (psensor->bisp0outputrequested != 1) {
                HW_NVMEDIA_LOG_ERR("The config of isp0 not enable, but register isp0 datacb!\r\n");
                return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_NOT_ENABLE_BUT_REGISTER_DATACB);
            }
            break;
        case HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP1:
            if (psensor->bisp1outputrequested != 1) {
                HW_NVMEDIA_LOG_ERR("The config of isp1 not enable, but register isp1 datacb!\r\n");
                return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_NOT_ENABLE_BUT_REGISTER_DATACB);
            }
            break;
        case HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP2:
            if (psensor->bisp2outputrequested != 1) {
                HW_NVMEDIA_LOG_ERR("The config of isp2 not enable, but register isp2 datacb!\r\n");
                return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_OUTPUTTYPE_NOT_ENABLE_BUT_REGISTER_DATACB);
            }
            break;
        default:
            RET_HW_RET_S32_COMMON_UNEXPECTED;
        }
        poutputpipelinecontext = (HWNvmediaOutputPipelineContext*)psensorpipeline_ops->parrayoutput[cbconfig.outputtype].priv;
        CHK_LOG_SENTENCE_HW_RET_S32(poutputpipelinecontext->RegisterDataCallback_ToOutput(pdatacbconfig, &cbconfig));
        bregistered[pdatacbconfig->type] = 1;
    }
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::IsQuit(u32* o_pisquit)
{
	*o_pisquit = bquit;
	return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::GetNotification(struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle,
	u32 i_timeoutus, HW_VIDEO_NOTIFRETSTATUS* o_pnotifretstatus)
{
    SIPLStatus status;
    size_t timeoutus = (size_t)((i_timeoutus == HW_TIMEOUT_FOREVER) ? HW_TIMEOUT_US_DEFAULT : i_timeoutus);
    NvSIPLPipelineNotifier::NotificationData notifdata;
    while (bquit == 0)
    {
        status = pnotificationqueue->Get(notifdata, timeoutus);
        if (status == NVSIPL_STATUS_OK) {
            if (i_benableinnerhandle)
            {
                CHK_LOG_SENTENCE_HW_RET_S32(notifinnerhandle(notifdata));
            }
            else
            {
                balwaysenableinnerhandle = 0;
            }
            CHK_LOG_SENTENCE_HW_RET_S32(notifconvert(notifdata, o_pnotification));
            *o_pnotifretstatus = HW_VIDEO_NOTIFRETSTATUS_GET;
            return 0;
        }
        else if (status == NVSIPL_STATUS_TIMED_OUT) {
            if (i_timeoutus != HW_TIMEOUT_FOREVER)
            {
                HW_NVMEDIA_LOG_TRACE("Queue timeout\r\n");
                *o_pnotifretstatus = HW_VIDEO_NOTIFRETSTATUS_TIMEOUT;
                return 0;
            }
            else
            {
                HW_NVMEDIA_LOG_DEBUG("Queue timeout\r\n");
                continue;
            }
        }
        else if (status == NVSIPL_STATUS_EOF) {
            HW_NVMEDIA_LOG_INFO("Queue shutdown\r\n");
            bquit = 1;
            *o_pnotifretstatus = HW_VIDEO_NOTIFRETSTATUS_QUIT;
            return 0;
        }
        else {
            HW_NVMEDIA_LOG_ERR("Unexpected queue return status\r\n");
            bquit = 1;
            *o_pnotifretstatus = HW_VIDEO_NOTIFRETSTATUS_QUIT;
            return 0;
        }
    }
    *o_pnotifretstatus = HW_VIDEO_NOTIFRETSTATUS_QUIT;
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::GetCount(u32* o_pcount)
{
    *o_pcount = (u32)pnotificationqueue->GetCount();
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::notifinnerhandle(NvSIPLPipelineNotifier::NotificationData& i_notifdata)
{
    switch (i_notifdata.eNotifType) {
    case NvSIPLPipelineNotifier::NOTIF_INFO_ICP_PROCESSING_DONE:
        HW_NVMEDIA_LOG_INFO("Pipeline: %u, NOTIF_INFO_ICP_PROCESSING_DONE\r\n", i_notifdata.uIndex);
        break;
    case NvSIPLPipelineNotifier::NOTIF_INFO_ISP_PROCESSING_DONE:
        HW_NVMEDIA_LOG_INFO("Pipeline: %u, NOTIF_INFO_ISP_PROCESSING_DONE\r\n", i_notifdata.uIndex);
        break;
    case NvSIPLPipelineNotifier::NOTIF_INFO_ACP_PROCESSING_DONE:
        HW_NVMEDIA_LOG_INFO("Pipeline: %u, NOTIF_INFO_ACP_PROCESSING_DONE\r\n", i_notifdata.uIndex);
        break;
    case NvSIPLPipelineNotifier::NOTIF_INFO_CDI_PROCESSING_DONE:
        HW_NVMEDIA_LOG_INFO("Pipeline: %u, NOTIF_INFO_CDI_PROCESSING_DONE\r\n", i_notifdata.uIndex);
        break;
    case NvSIPLPipelineNotifier::NOTIF_WARN_ICP_FRAME_DROP:
        HW_NVMEDIA_LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_FRAME_DROP\r\n", i_notifdata.uIndex);
        m_uNumFrameDrops++;
        break;
    case NvSIPLPipelineNotifier::NOTIF_WARN_ICP_FRAME_DISCONTINUITY:
        HW_NVMEDIA_LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_FRAME_DISCONTINUITY\r\n", i_notifdata.uIndex);
        break;
    case NvSIPLPipelineNotifier::NOTIF_WARN_ICP_CAPTURE_TIMEOUT:
        HW_NVMEDIA_LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_CAPTURE_TIMEOUT\r\n", i_notifdata.uIndex);
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_ICP_BAD_INPUT_STREAM:
        HW_NVMEDIA_LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_BAD_INPUT_STREAM\r\n", i_notifdata.uIndex);
        if (!bignoreerror) {
            m_bInError = true; // Treat this as fatal error only if link recovery is not enabled.
        }
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_ICP_CAPTURE_FAILURE:
        HW_NVMEDIA_LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_CAPTURE_FAILURE\r\n", i_notifdata.uIndex);
        m_bInError = true;
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE:
        HW_NVMEDIA_LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE\r\n", i_notifdata.uIndex);
        m_bInError = true;
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_ISP_PROCESSING_FAILURE:
        HW_NVMEDIA_LOG_ERR("Pipeline: %u, NOTIF_ERROR_ISP_PROCESSING_FAILURE\r\n", i_notifdata.uIndex);
        m_bInError = true;
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_ACP_PROCESSING_FAILURE:
        HW_NVMEDIA_LOG_ERR("Pipeline: %u, NOTIF_ERROR_ACP_PROCESSING_FAILURE\r\n", i_notifdata.uIndex);
        m_bInError = true;
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE:
        HW_NVMEDIA_LOG_ERR("Pipeline: %u, NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE\r\n", i_notifdata.uIndex);
        if (!bignoreerror) {
            m_bInError = true; // Treat this as fatal error only if link recovery is not enabled.
        }
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_INTERNAL_FAILURE:
        HW_NVMEDIA_LOG_ERR("Pipeline: %u, NOTIF_ERROR_INTERNAL_FAILURE\r\n", i_notifdata.uIndex);
        m_bInError = true;
        break;
    default:
        HW_NVMEDIA_LOG_WARN("Pipeline: %u, Unknown/Invalid notification\r\n", i_notifdata.uIndex);
        break;
    }
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::notifconvert(NvSIPLPipelineNotifier::NotificationData& i_notifdata, struct hw_video_notification_t* o_pnotification)
{
    o_pnotification->notiftype = (HW_VIDEO_NOTIFTYPE)i_notifdata.eNotifType;
    o_pnotification->blockindex = psensorpipeline_ops->blockindex;
    o_pnotification->sensorindex = psensorpipeline_ops->sensorindex;
    o_pnotification->framecapturetsc = i_notifdata.frameCaptureTSC;
    o_pnotification->framecapturestarttsc = i_notifdata.frameCaptureStartTSC;
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::ResetEnableInnerHandle()
{
    balwaysenableinnerhandle = 1;
    HW_NVMEDIA_LOG_UNMASK("Reset balwaysenableinnerhandle to 1!\r\n");
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::IsSensorInError(u32* o_pissensorinerror)
{
    if (balwaysenableinnerhandle == 0)
    {
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_SENSORPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE);
    }
    *o_pissensorinerror = m_bInError;
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::ResetFrameDropCounter()
{
    m_uNumFrameDrops = 0;
    return 0;
}

hw_ret_s32 HWNvmediaSensorPipelineContext::GetNumFrameDrops(u32* o_pnumframedrops)
{
    *o_pnumframedrops = m_uNumFrameDrops;
    return 0;
}

static s32 hw_nvmedia_sensorpipeline_isquit(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, u32* o_pisquit)
{
    HWNvmediaSensorPipelineContext* pnvmediasensorpipeline = (HWNvmediaSensorPipelineContext*)i_psensorpipeline_ops->priv;
    return pnvmediasensorpipeline->IsQuit(o_pisquit);
}

static s32 hw_nvmedia_sensorpipeline_getnotification(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops,
    struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle,
    u32 i_timeoutus, HW_VIDEO_NOTIFRETSTATUS* o_pnotifretstatus)
{
    HWNvmediaSensorPipelineContext* pnvmediasensorpipeline = (HWNvmediaSensorPipelineContext*)i_psensorpipeline_ops->priv;
    return pnvmediasensorpipeline->GetNotification(o_pnotification, i_benableinnerhandle, i_timeoutus, o_pnotifretstatus);
}

static s32 hw_nvmedia_sensorpipeline_getcount(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops,
    u32* o_pcount)
{
    HWNvmediaSensorPipelineContext* pnvmediasensorpipeline = (HWNvmediaSensorPipelineContext*)i_psensorpipeline_ops->priv;
    return pnvmediasensorpipeline->GetCount(o_pcount);
}

static s32 hw_nvmedia_sensorpipeline_resetenableinnerhandle(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops)
{
    HWNvmediaSensorPipelineContext* pnvmediasensorpipeline = (HWNvmediaSensorPipelineContext*)i_psensorpipeline_ops->priv;
    return pnvmediasensorpipeline->ResetEnableInnerHandle();
}

static s32 hw_nvmedia_sensorpipeline_issensorinerror(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops,
    u32* o_pissensorinerror)
{
    HWNvmediaSensorPipelineContext* pnvmediasensorpipeline = (HWNvmediaSensorPipelineContext*)i_psensorpipeline_ops->priv;
    return pnvmediasensorpipeline->IsSensorInError(o_pissensorinerror);
}

static s32 hw_nvmedia_sensorpipeline_resetframedropcounter(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops)
{
    HWNvmediaSensorPipelineContext* pnvmediasensorpipeline = (HWNvmediaSensorPipelineContext*)i_psensorpipeline_ops->priv;
    return pnvmediasensorpipeline->ResetFrameDropCounter();
}

static s32 hw_nvmedia_sensorpipeline_getnumframedrops(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops,
    u32* o_pnumframedrops)
{
    HWNvmediaSensorPipelineContext* pnvmediasensorpipeline = (HWNvmediaSensorPipelineContext*)i_psensorpipeline_ops->priv;
    return pnvmediasensorpipeline->GetNumFrameDrops(o_pnumframedrops);
}

s32 hw_nvmedia_sensorpipeline_setops(struct hw_video_sensorpipeline_ops_t* io_psensorpipeline_ops)
{
    io_psensorpipeline_ops->isquit = hw_nvmedia_sensorpipeline_isquit;
    io_psensorpipeline_ops->getnotification = hw_nvmedia_sensorpipeline_getnotification;
    io_psensorpipeline_ops->getcount = hw_nvmedia_sensorpipeline_getcount;
    io_psensorpipeline_ops->notif_ops.resetenableinnerhandle = hw_nvmedia_sensorpipeline_resetenableinnerhandle;
    io_psensorpipeline_ops->notif_ops.issensorinerror = hw_nvmedia_sensorpipeline_issensorinerror;
    io_psensorpipeline_ops->notif_ops.resetframedropcounter = hw_nvmedia_sensorpipeline_resetframedropcounter;
    io_psensorpipeline_ops->notif_ops.getnumframedrops = hw_nvmedia_sensorpipeline_getnumframedrops;
    return 0;
}
