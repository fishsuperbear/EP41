#ifndef HW_NVMEDIA_PIPELINE_IMPL_H
#define HW_NVMEDIA_PIPELINE_IMPL_H

#include "hw_nvmedia_device_single_process_impl.h"

/*
* Define the priv of struct hw_video_blockpipeline_ops_t here.
*/
class HWNvmediaBlockPipelineContext
{
private:
	HWNvmediaBlockPipelineContext();
public:
	HWNvmediaBlockPipelineContext(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, HWNvmediaContext* i_pcontext)
	{
		pblockpipeline_ops = i_pblockpipeline_ops;
		pcontext = i_pcontext;
	}
	~HWNvmediaBlockPipelineContext()
	{
		hw_plat_mutex_deinit(&mutexdeviceblocklog);
	}
public:
	// called immediately after constructor.
	hw_ret_s32 Init();
public:
	hw_ret_s32 IsQuit(u32* o_pisquit);
	hw_ret_s32 GetNotification(struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle, 
		u32 i_timeoutus, HW_VIDEO_NOTIFRETSTATUS* o_pnotifretstatus);
	hw_ret_s32 GetCount(u32* o_pcount);
private:
	bool istruegpiointerrupt(const uint32_t* gpioIdxs, uint32_t numGpioIdxs);
	// recursive lock
	void deviceblocklog_lock()
	{
		hw_plat_mutex_lock(&mutexdeviceblocklog);
	}
	void deviceblocklog_unlock()
	{
		hw_plat_mutex_unlock(&mutexdeviceblocklog);
	}
	void handledeserializererror();
	void handlecameramoduleerror(u32 i_index);
protected:
	hw_ret_s32 notifinnerhandle(NvSIPLPipelineNotifier::NotificationData& i_notifdata);
	// convert the NvSIPLPipelineNotifier::NotificationData to struct hw_video_notification_t.
	hw_ret_s32 notifconvert(NvSIPLPipelineNotifier::NotificationData& i_notifdata, struct hw_video_notification_t* o_pnotification);
public:
	// reset the balwaysenableinnerhandle to 1, do not need to call it when init
	hw_ret_s32 ResetEnableInnerHandle();
	hw_ret_s32 IsBlockInError(u32* o_pisblockinerror);
	hw_ret_s32 GetSiplErrorDetails_Deserializer(u8** o_pperrorbuffer, u32* o_psizewritten);
	hw_ret_s32 GetSiplErrorDetails_Serializer(u8** o_pperrorbuffer, u32* o_psizewritten);
	hw_ret_s32 GetSiplErrorDetails_Sensor(u8** o_pperrorbuffer, u32* o_psizewritten);
public:
	struct hw_video_blockpipeline_ops_t*	pblockpipeline_ops;
	struct HWNvmediaContext*				pcontext;
public:
	// notificationqueue for device block
	INvSIPLNotificationQueue*				pnotificationqueue;
	/*
	* 0 or 1
	* 0 mean has at least once not set i_benableinnerhandle to 1 when call getnotification
	* of hw_video_blockpipeline_ops_t.
	*/
	u32										balwaysenableinnerhandle = 1U;
	/*
	* 0 or 1
	* 1 mean has received NVSIPL_STATUS_EOF when get notif.
	* 0 mean has not received NVSIPL_STATUS_EOF when get notif.
	*/
	u32										bquit = 0U;
public:
	// recursive lock
	hw_mutex_t								mutexdeviceblocklog;
	// protected by mutexdeviceblocklog
	u32										deviceblocklogindex = 0;
	// see bIgnoreError in nvsipl_multicast sample
	u32										bignoreerror = 0;
	size_t									m_uErrorSize{};
	SIPLErrorDetails						m_oDeserializerErrorInfo{};
	SIPLErrorDetails						m_oSerializerErrorInfo{};
	SIPLErrorDetails						m_oSensorErrorInfo{};
	u32										m_bInError = 0U;
};

s32 hw_nvmedia_blockpipeline_setops(struct hw_video_blockpipeline_ops_t* io_pblockpipeline_ops);

/*
* Define the priv of struct hw_video_sensorpipeline_ops_t here.
*/
class HWNvmediaSensorPipelineContext
{
private:
	HWNvmediaSensorPipelineContext();
public:
	HWNvmediaSensorPipelineContext(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, HWNvmediaContext* i_pcontext)
	{
		psensorpipeline_ops = i_psensorpipeline_ops;
		pcontext = i_pcontext;
	}
	~HWNvmediaSensorPipelineContext()
	{

	}
public:
	// called immediately after constructor.
	hw_ret_s32 Init();
	// called after every output type path has init
	hw_ret_s32 RegisterDataCallback();
public:
	hw_ret_s32 IsQuit(u32* o_pisquit);
	hw_ret_s32 GetNotification(struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle, 
		u32 i_timeoutus, HW_VIDEO_NOTIFRETSTATUS* o_pnotifretstatus);
	hw_ret_s32 GetCount(u32* o_pcount);
protected:
	hw_ret_s32 notifinnerhandle(NvSIPLPipelineNotifier::NotificationData& i_notifdata);
	// convert the NvSIPLPipelineNotifier::NotificationData to struct hw_video_notification_t.
	hw_ret_s32 notifconvert(NvSIPLPipelineNotifier::NotificationData& i_notifdata, struct hw_video_notification_t* o_pnotification);
public:
	// reset the balwaysenableinnerhandle to 1, do not need to call it when init
	hw_ret_s32 ResetEnableInnerHandle();
	hw_ret_s32 IsSensorInError(u32* o_pissensorinerror);
	hw_ret_s32 ResetFrameDropCounter();
	hw_ret_s32 GetNumFrameDrops(u32* o_pnumframedrops);
public:
	struct hw_video_sensorpipeline_ops_t*	psensorpipeline_ops;
	struct HWNvmediaContext*				pcontext;
public:
	// notificationqueue for sensor pipeline, equal to notificationQueue of NvSIPLPipelineQueues pipelinequeue.
	INvSIPLNotificationQueue*				pnotificationqueue;
	NvSIPLPipelineQueues					pipelinequeue;
	/*
	* 0 or 1
	* 0 mean has at least once not set i_benableinnerhandle to 1 when call getnotification
	* of hw_video_sensorpipeline_ops_t.
	*/
	u32										balwaysenableinnerhandle = 1U;
	/*
	* 0 or 1
	* 1 mean has received NVSIPL_STATUS_EOF when get notif.
	* 0 mean has not received NVSIPL_STATUS_EOF when get notif.
	*/
	u32										bquit = 0U;
public:
	// see bIgnoreError in nvsipl_multicast sample
	u32										bignoreerror = 0;
	u32										m_uNumFrameDrops = 0U;
	u32										m_bInError = 0U;
};

s32 hw_nvmedia_sensorpipeline_setops(struct hw_video_sensorpipeline_ops_t* io_psensorpipeline_ops);

#endif
