#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_log_deviceblock_impl.h"

hw_ret_s32 HWNvmediaBlockPipelineContext::Init()
{
    CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(pcontext->_pcamera->GetMaxErrorSize(pblockpipeline_ops->blockindex, m_uErrorSize),
        "GetMaxErrorSize");
    hw_plat_mutex_init(&mutexdeviceblocklog, HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE);
    if (m_uErrorSize != 0U)
    {
        m_oDeserializerErrorInfo.upErrorBuffer.reset(new uint8_t[m_uErrorSize]);
        m_oDeserializerErrorInfo.bufferSize = m_uErrorSize;

        m_oSerializerErrorInfo.upErrorBuffer.reset(new uint8_t[m_uErrorSize]);
        m_oSerializerErrorInfo.bufferSize = m_uErrorSize;

        m_oSensorErrorInfo.upErrorBuffer.reset(new uint8_t[m_uErrorSize]);
        m_oSensorErrorInfo.bufferSize = m_uErrorSize;
    }
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::IsQuit(u32* o_pisquit)
{
    *o_pisquit = bquit;
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::GetNotification(struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle,
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

hw_ret_s32 HWNvmediaBlockPipelineContext::GetCount(u32* o_pcount)
{
    *o_pcount = (u32)pnotificationqueue->GetCount();
    return 0;
}

bool HWNvmediaBlockPipelineContext::istruegpiointerrupt(const uint32_t* gpioIdxs, uint32_t numGpioIdxs)
{
    /*
    * Get disambiguated GPIO interrupt event codes, to determine whether
    * true interrupts or propagation functionality fault occurred.
    */

    bool true_interrupt = false;
    SIPLGpioEvent code;
    SIPLStatus status;

    for (uint32_t i = 0U; i < numGpioIdxs; i++)
    {
        status = pcontext->_pcamera->GetErrorGPIOEventInfo(pblockpipeline_ops->blockindex,
            gpioIdxs[i],
            code);
        if (status == NVSIPL_STATUS_NOT_SUPPORTED) {
            HW_NVMEDIA_LOG_INFO("GetErrorGPIOEventInfo is not supported by OS backend currently!\r\n");
            /* Allow app to fetch detailed error info, same as in case of true interrupt. */
            return true;
        }
        else if (status != NVSIPL_STATUS_OK) {
            HW_NVMEDIA_LOG_ERR("DeviceBlock: %u, GetErrorGPIOEventInfo failed\r\n", pblockpipeline_ops->blockindex);
            m_bInError = 1;
            return false;
        }

        /*
         * If no error condition code is returned, and at least one GPIO has
         * NVSIPL_GPIO_EVENT_INTR status, return true.
         */
        if (code == NVSIPL_GPIO_EVENT_INTR) {
            true_interrupt = true;
        }
        else if (code != NVSIPL_GPIO_EVENT_NOTHING) {
            // GPIO functionality fault (treat as fatal)
            m_bInError = 1;
            return false;
        }
    }

    return true_interrupt;
}

void HWNvmediaBlockPipelineContext::handledeserializererror()
{
    bool isRemoteError{ false };
    uint8_t linkErrorMask{ 0U };
    DeviceBlockInfo* pdeviceblockinfo = &pcontext->_platformcfg.deviceBlockList[pblockpipeline_ops->blockindex];

    /* Get detailed error information (if error size is non-zero) and
     * information about remote error and link error. */
    SIPLStatus status = pcontext->_pcamera->GetDeserializerErrorInfo(
        pblockpipeline_ops->blockindex,
        (m_uErrorSize > 0) ? &m_oDeserializerErrorInfo : nullptr,
        isRemoteError,
        linkErrorMask);
    if (status != NVSIPL_STATUS_OK) {
        HW_NVMEDIA_LOG_ERR("DeviceBlock: %u, GetDeserializerErrorInfo failed\n", pblockpipeline_ops->blockindex);
        m_bInError = 1;
        return;
    }

    if ((m_uErrorSize > 0) && (m_oDeserializerErrorInfo.sizeWritten != 0)) {
#if (HW_NVMEDIA_MAY_CHANGE_ME_LATER == 1)
        /*
        * We may change the log output way to use hw_plat_logblockoutput implementation.
        * Now we add lock and output to a seperate file.
        */
        HW_NVMEDIA_LOG_INFO("DeviceBlock[%u] Deserializer Error Buffer[index:%u] output in seperate file!\r\n",
            pblockpipeline_ops->blockindex, deviceblocklogindex);
        deviceblocklog_lock();
        HW_NVMEDIA_DEVICEBLOCK_LOG_INFO("DeviceBlock[%u] Deserializer Error Buffer[index:%u]: ",
            pblockpipeline_ops->blockindex, deviceblocklogindex);
        deviceblocklogindex++;
        for (uint32_t i = 0; i < m_oDeserializerErrorInfo.sizeWritten; i++) {
            HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_INFO("%c ", m_oDeserializerErrorInfo.upErrorBuffer[i]);
        }
        HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_INFO("\r\n");
        deviceblocklog_unlock();
#endif
        m_bInError = 1;
    }

    if (isRemoteError) {
#if (HW_NVMEDIA_MAY_CHANGE_ME_LATER == 1)
        /*
        * We may change the log output way to use hw_plat_logblockoutput implementation.
        * Now we add lock and output to a seperate file.
        */
        HW_NVMEDIA_LOG_INFO("DeviceBlock[%u] Deserializer Remote Error[index:%u] output in seperate file!\r\n",
            pblockpipeline_ops->blockindex, deviceblocklogindex);
        deviceblocklog_lock();
        HW_NVMEDIA_DEVICEBLOCK_LOG_INFO("DeviceBlock[%u] Deserializer Remote Error[index:%u].\r\n",
            pblockpipeline_ops->blockindex, deviceblocklogindex);
        deviceblocklogindex++;
        for (uint32_t i = 0; i < pdeviceblockinfo->numCameraModules; i++) {
            handlecameramoduleerror(pdeviceblockinfo->cameraModuleInfoList[i].sensorInfo.id);
        }
        deviceblocklog_unlock();
#endif
    }

    if (linkErrorMask != 0U) {
        HW_NVMEDIA_LOG_ERR("DeviceBlock: %u, Deserializer link error. mask: %u\n", pblockpipeline_ops->blockindex, linkErrorMask);
        m_bInError = 1;
    }
}

void HWNvmediaBlockPipelineContext::handlecameramoduleerror(u32 i_index)
{
#if (HW_NVMEDIA_MAY_CHANGE_ME_LATER == 1)
    deviceblocklog_lock();
    if (m_uErrorSize > 0) {
        /* Get detailed error information. */
        SIPLStatus status = pcontext->_pcamera->GetModuleErrorInfo(
            i_index,
            &m_oSerializerErrorInfo,
            &m_oSensorErrorInfo);
        if (status != NVSIPL_STATUS_OK) {
            HW_NVMEDIA_LOG_ERR("index: %u, GetModuleErrorInfo failed\r\n", i_index);
            m_bInError = true;
            return;
        }

        if (m_oSerializerErrorInfo.sizeWritten != 0) {
            HW_NVMEDIA_DEVICEBLOCK_LOG_INFO("Pipeline[%u] Serializer Error Buffer: ",
                i_index);
            for (uint32_t i = 0; i < m_oSerializerErrorInfo.sizeWritten; i++) {
                HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_INFO("%c ", m_oSerializerErrorInfo.upErrorBuffer[i]);
            }
            HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_INFO("\r\n");
            m_bInError = true;
        }

        if (m_oSensorErrorInfo.sizeWritten != 0) {
            HW_NVMEDIA_DEVICEBLOCK_LOG_INFO("Pipeline[%u] Sensor Error Buffer: ", i_index);
            for (uint32_t i = 0; i < m_oSensorErrorInfo.sizeWritten; i++) {
                HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_INFO("%c ", m_oSensorErrorInfo.upErrorBuffer[i]);
            }
            HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_INFO("\r\n");
            m_bInError = true;
        }
    }
    deviceblocklog_unlock();
#endif
}

hw_ret_s32 HWNvmediaBlockPipelineContext::notifinnerhandle(NvSIPLPipelineNotifier::NotificationData& i_notifdata)
{
    DeviceBlockInfo* pdeviceblockinfo = &pcontext->_platformcfg.deviceBlockList[pblockpipeline_ops->blockindex];
    switch (i_notifdata.eNotifType) {
    case NvSIPLPipelineNotifier::NOTIF_ERROR_DESERIALIZER_FAILURE:
        HW_NVMEDIA_LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_DESERIALIZER_FAILURE\r\n", pblockpipeline_ops->blockindex);
        if (!bignoreerror) {
            if (istruegpiointerrupt(i_notifdata.gpioIdxs, i_notifdata.numGpioIdxs)) {
                handledeserializererror();
            }
        }
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_SERIALIZER_FAILURE:
        HW_NVMEDIA_LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_SERIALIZER_FAILURE\r\n", pblockpipeline_ops->blockindex);
        if (!bignoreerror) {
            for (uint32_t i = 0; i < pblockpipeline_ops->numsensors; i++) {
                if ((i_notifdata.uLinkMask & (1 << (pdeviceblockinfo->cameraModuleInfoList[i].linkIndex))) != 0) {
                    if (istruegpiointerrupt(i_notifdata.gpioIdxs, i_notifdata.numGpioIdxs)) {
                        handlecameramoduleerror(pdeviceblockinfo->cameraModuleInfoList[i].sensorInfo.id);
                    }
                }
            }
        }
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_SENSOR_FAILURE:
        HW_NVMEDIA_LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_SENSOR_FAILURE\r\n", pblockpipeline_ops->blockindex);
        if (!bignoreerror) {
            for (uint32_t i = 0; i < pblockpipeline_ops->numsensors; i++) {
                if ((i_notifdata.uLinkMask & (1 << (pdeviceblockinfo->cameraModuleInfoList[i].linkIndex))) != 0) {
                    if (istruegpiointerrupt(i_notifdata.gpioIdxs, i_notifdata.numGpioIdxs)) {
                        handlecameramoduleerror(pdeviceblockinfo->cameraModuleInfoList[i].sensorInfo.id);
                    }
                }
            }
        }
        break;
    case NvSIPLPipelineNotifier::NOTIF_ERROR_INTERNAL_FAILURE:
        HW_NVMEDIA_LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_INTERNAL_FAILURE\r\n", pblockpipeline_ops->blockindex);
        m_bInError = 1;
        break;
    default:
        HW_NVMEDIA_LOG_WARN("DeviceBlock: %u, Unknown/Invalid notification\r\n", pblockpipeline_ops->blockindex);
        break;
    }
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::notifconvert(NvSIPLPipelineNotifier::NotificationData& i_notifdata, struct hw_video_notification_t* o_pnotification)
{
    o_pnotification->notiftype = (HW_VIDEO_NOTIFTYPE)i_notifdata.eNotifType;
    o_pnotification->blockindex = pblockpipeline_ops->blockindex;
    o_pnotification->sensorindex = -1;  // not valid
    o_pnotification->framecapturetsc = i_notifdata.frameCaptureTSC;
    o_pnotification->framecapturestarttsc = i_notifdata.frameCaptureStartTSC;
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::ResetEnableInnerHandle()
{
    balwaysenableinnerhandle = 1;
    HW_NVMEDIA_LOG_UNMASK("Reset balwaysenableinnerhandle to 1!\r\n");
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::IsBlockInError(u32* o_pisblockinerror)
{
    if (balwaysenableinnerhandle == 0)
    {
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_BLOCKPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE);
    }
    *o_pisblockinerror = m_bInError;
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::GetSiplErrorDetails_Deserializer(u8** o_pperrorbuffer, u32* o_psizewritten)
{
    if (balwaysenableinnerhandle == 0)
    {
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_BLOCKPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE);
    }
    *o_pperrorbuffer = m_oDeserializerErrorInfo.upErrorBuffer.get();
    *o_psizewritten = (u32)m_oDeserializerErrorInfo.sizeWritten;
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::GetSiplErrorDetails_Serializer(u8** o_pperrorbuffer, u32* o_psizewritten)
{
    if (balwaysenableinnerhandle == 0)
    {
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_BLOCKPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE);
    }
    *o_pperrorbuffer = m_oSerializerErrorInfo.upErrorBuffer.get();
    *o_psizewritten = (u32)m_oSerializerErrorInfo.sizeWritten;
    return 0;
}

hw_ret_s32 HWNvmediaBlockPipelineContext::GetSiplErrorDetails_Sensor(u8** o_pperrorbuffer, u32* o_psizewritten)
{
    if (balwaysenableinnerhandle == 0)
    {
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_BLOCKPIPELINE_NOT_ALWAYS_ENABLE_INNERHANDLE);
    }
    *o_pperrorbuffer = m_oSensorErrorInfo.upErrorBuffer.get();
    *o_psizewritten = (u32)m_oSensorErrorInfo.sizeWritten;
    return 0;
}

static s32 hw_nvmedia_blockpipeline_isquit(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, u32* o_pisquit)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->IsQuit(o_pisquit);
}

static s32 hw_nvmedia_blockpipeline_getnotification(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops,
    struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle,
    u32 i_timeoutus, HW_VIDEO_NOTIFRETSTATUS* o_pnotifretstatus)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->GetNotification(o_pnotification, i_benableinnerhandle, i_timeoutus, o_pnotifretstatus);
}

static s32 hw_nvmedia_blockpipeline_getcount(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops,
    u32* o_pcount)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->GetCount(o_pcount);
}

static s32 hw_nvmedia_blockpipeline_resetenableinnerhandle(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->ResetEnableInnerHandle();
}

static s32 hw_nvmedia_blockpipeline_isblockinerror(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops,
    u32* o_pisblockinerror)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->IsBlockInError(o_pisblockinerror);
}

static s32 hw_nvmedia_blockpipeline_getsiplerrordetails_deserializer(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops,
    u8** o_pperrorbuffer, u32* o_psizewritten)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->GetSiplErrorDetails_Deserializer(o_pperrorbuffer, o_psizewritten);
}

static s32 hw_nvmedia_blockpipeline_getsiplerrordetails_serializer(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops,
    u8** o_pperrorbuffer, u32* o_psizewritten)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->GetSiplErrorDetails_Serializer(o_pperrorbuffer, o_psizewritten);
}

static s32 hw_nvmedia_blockpipeline_getsiplerrordetails_sensor(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops,
    u8** o_pperrorbuffer, u32* o_psizewritten)
{
    HWNvmediaBlockPipelineContext* pnvmediablockpipeline = (HWNvmediaBlockPipelineContext*)i_pblockpipeline_ops->priv;
    return pnvmediablockpipeline->GetSiplErrorDetails_Sensor(o_pperrorbuffer, o_psizewritten);
}

s32 hw_nvmedia_blockpipeline_setops(struct hw_video_blockpipeline_ops_t* io_pblockpipeline_ops)
{
    io_pblockpipeline_ops->isquit = hw_nvmedia_blockpipeline_isquit;
    io_pblockpipeline_ops->getnotification = hw_nvmedia_blockpipeline_getnotification;
    io_pblockpipeline_ops->getcount = hw_nvmedia_blockpipeline_getcount;
    io_pblockpipeline_ops->notif_ops.resetenableinnerhandle = hw_nvmedia_blockpipeline_resetenableinnerhandle;
    io_pblockpipeline_ops->notif_ops.isblockinerror = hw_nvmedia_blockpipeline_isblockinerror;
    io_pblockpipeline_ops->notif_ops.getsiplerrordetails_deserializer = hw_nvmedia_blockpipeline_getsiplerrordetails_deserializer;
    io_pblockpipeline_ops->notif_ops.getsiplerrordetails_serializer = hw_nvmedia_blockpipeline_getsiplerrordetails_serializer;
    io_pblockpipeline_ops->notif_ops.getsiplerrordetails_sensor = hw_nvmedia_blockpipeline_getsiplerrordetails_sensor;
    return 0;
}
