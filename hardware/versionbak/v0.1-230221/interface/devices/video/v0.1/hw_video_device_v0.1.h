#ifndef HW_VIDEO_DEVICE_V0_1_H
#define HW_VIDEO_DEVICE_V0_1_H

#include "hw_video_version.h"

__BEGIN_DECLS

#define HW_VIDEO_NUMBLOCKS_MAX			4
#define HW_VIDEO_NUMSENSORS_PER_BLOCK	4

enum HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE
{
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MIN = 0,
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MINMINUSONE = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MIN - 1,

	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP,
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0,
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP1,
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP2,

	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MAXADDONE,
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MAX = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MAXADDONE - 1,

	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_COUNT = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MAX - HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MIN + 1,
};

/*
* The callback type of which we need to register the data.
*/
enum HW_VIDEO_REGDATACB_TYPE
{
	HW_VIDEO_REGDATACB_TYPE_MIN = 0,
	HW_VIDEO_REGDATACB_TYPE_MINMINUSONE = HW_VIDEO_REGDATACB_TYPE_MIN - 1,

	HW_VIDEO_REGDATACB_TYPE_RAW12,
	// properly not use it
	HW_VIDEO_REGDATACB_TYPE_YUV422,
	HW_VIDEO_REGDATACB_TYPE_YUV420,	// common yuv420 type
	HW_VIDEO_REGDATACB_TYPE_YUV420_PRIV,	// private yuv420 type
	HW_VIDEO_REGDATACB_TYPE_RGBA,
	HW_VIDEO_REGDATACB_TYPE_CUDA,

	HW_VIDEO_REGDATACB_TYPE_MAXADDONE,
	HW_VIDEO_REGDATACB_TYPE_MAX = HW_VIDEO_REGDATACB_TYPE_MAXADDONE - 1,

	HW_VIDEO_REGDATACB_TYPE_COUNT = HW_VIDEO_REGDATACB_TYPE_MAX - HW_VIDEO_REGDATACB_TYPE_MIN + 1,
};

enum HW_VIDEO_BUFFERFORMAT_MAINTYPE
{
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_MIN = 0,
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_MINMINUSONE = HW_VIDEO_BUFFERFORMAT_MAINTYPE_MIN - 1,
	
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_RAW12,
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_LUMA,
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV422,
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV420,
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_RGBA,

	HW_VIDEO_BUFFERFORMAT_MAINTYPE_MAXADDONE,
	HW_VIDEO_BUFFERFORMAT_MAINTYPE_MAX = HW_VIDEO_BUFFERFORMAT_MAINTYPE_MAXADDONE - 1,
};

enum HW_VIDEO_BUFFERFORMAT_SUBTYPE
{
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_MIN = 0,
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_MINMINUSONE = HW_VIDEO_BUFFERFORMAT_SUBTYPE_MIN - 1,

	HW_VIDEO_BUFFERFORMAT_SUBTYPE_RAW12,
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_LUMA,
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV422,
	// common yuv420 format
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PL,
	// nvidia private yuv420 format
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_BL,
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_RGBA,

	HW_VIDEO_BUFFERFORMAT_SUBTYPE_MAXADDONE,
	HW_VIDEO_BUFFERFORMAT_SUBTYPE_MAX = HW_VIDEO_BUFFERFORMAT_SUBTYPE_MAXADDONE - 1,
};

typedef void* hw_video_handlepipeline;

struct hw_video_bufferobj_t;
struct hw_video_outputpipeline_ops_t;
struct hw_video_t;

#define HW_VIDEO_BUFFER_NUMSURFACES_MAX		2

/*
* The buffer information is retrieved to input parameter when datacb function is called.
*/
typedef struct hw_video_bufferinfo_t
{
	HW_VIDEO_REGDATACB_TYPE				regdatacbtype;
	// begin from 0
	u32									blockindex;
	// begin from 0
	u32									sensorindex;
	HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE	outputtype;
	/*
	* See HW_VIDEO_BUFFERFORMAT_MAINTYPE.
	* Valid in most situation.
	* May not be valid when regdatacbtype is type like cuda etc..
	*/
	u32									format_maintype;
	/*
	* See HW_VIDEO_BUFFERFORMAT_SUBTYPE
	* Valid in most situation.
	* May not be valid when regdatacbtype is type like cuda etc..
	*/
	u32									format_subtype;
	// valid when maintype is YUV XXX
	u32									numplane;
	// the whole data buffer info, may not be valid when some type like cuda
	void*								pbuff;
	// may not be valid when some type like cuda
	u32									size;
	// same as the input when you call pipeline_open
	u32									bsynccb;
	/*
	* 1 mean you need to free after use, 0 mean not
	* Properly when you set async data cb mode, bneedfree is 1.
	*/
	u32									bneedfree;
	/*
	* custom pointer, the same as the pointer set to pipeline_open specific path
	*/
	void*								pcustom;
} hw_video_bufferinfo_t;

typedef void* hw_video_handleframe;

enum HW_VIDEO_FRAMERETSTATUS
{
	HW_VIDEO_FRAMERETSTATUS_MIN = 0,
	HW_VIDEO_FRAMERETSTATUS_MINMINUSONE = HW_VIDEO_FRAMERETSTATUS_MIN - 1,

	// successfully get the frame handle
	HW_VIDEO_FRAMERETSTATUS_GET,
	HW_VIDEO_FRAMERETSTATUS_TIMEOUT,
	HW_VIDEO_FRAMERETSTATUS_QUIT,

	HW_VIDEO_FRAMERETSTATUS_MAXADDONE,
	HW_VIDEO_FRAMERETSTATUS_MAX = HW_VIDEO_FRAMERETSTATUS_MAXADDONE - 1,
};

typedef struct hw_video_outputpipeline_frame_ops_t
{
	/*
	* Do not need to call it when init.
	* When you want to get the correct frame count gap from last get, you should not call 
	* resetframecounter after getframecount, because the two operation is not atomic.
	* You should call getframecount every time you get and calculate the frame count gap by 
	* yourself to get the correct frame count gap.
	*/
	s32(*resetframecounter)(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops);
	/*
	* Output to *o_pframecount, the total frame count of the specific output.
	* You may create more than one thread to handle the frame, it will sum up all of the frame
	* the specific output path.
	* The count of frame which is handled.
	*/
	s32(*getframecount)(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops, u64* o_pframecount);
} hw_video_outputpipeline_frame_ops_t;

typedef struct hw_video_outputpipeline_ops_t
{
	/*
	* 0 or 1
	* Whether the specific sensor pipeline is enabled or not.
	*/
	u32											bused;
	u32											blockindex;	// valid when bused is 1
	u32											sensorindex;	// valid when bused is 1
	u32											outputtype;	// Valid when bused is 1.
	
	void*										priv;	// internal use only
	
	/*
	* The timeout unit is us.
	* When you need to wait forever, use HW_TIMEOUT_FOREVER define, it is -1 properly.
	* When function successfully return, *o_phandleframe will be set to the notification received.
	* The function will output the handle of frame just get to *o_phandleframe which you can use 
	* it to call handle function to trigger the frame handle pipeline.
	* Properly return 0, when return 0, *o_pframeretstatus output value is valid.
	* o_pframeretstatus should not be NULL, *o_pframeretstatus output HW_VIDEO_FRAMERETSTATUS.
	*/
	s32(*gethandleframe)(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
		hw_video_handleframe* o_phandleframe, u32 i_timeoutus,
		HW_VIDEO_FRAMERETSTATUS* o_pframeretstatus);
	/*
	* When call the handle function, it frame count will plus one.
	* You can use resetframecounter to reset the frame count.
	*/
	s32(*handle)(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
		hw_video_handleframe i_handleframe);
	/*
	* When call the skiphandle function, it frame count will not plus one.
	* When you successfully get the frame handle by gethandleframe function, you should use
	* the skiphandle to skip the handle when you do not need to handle the frame.
	*/
	s32(*skiphandle)(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
		hw_video_handleframe i_handleframe);
	/*
	* BE CAREFUL! IT IS NOT FRAME COUNT! Use funtion of frame_ops to get frame count.
	* Get the elements count currently in the specific output type pipeline queue.
	* Output the count to *o_pcount.
	*/
	s32(*getcount)(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops, u32* o_pcount);

	/*
	* Always valid.
	*/
	struct hw_video_outputpipeline_frame_ops_t	frame_ops;
} hw_video_outputpipeline_ops_t;

enum HW_VIDEO_NOTIFTYPE
{
	/**
	* Pipeline event, indicates ICP processing is finished.
	* @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
	*/
	NOTIF_INFO_ICP_PROCESSING_DONE = 0,

	/**
	* Pipeline event, indicates ISP processing is finished.
	* @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
	*/
	NOTIF_INFO_ISP_PROCESSING_DONE = 1,

	/**
	* Pipeline event, indicates auto control processing is finished.
	* @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
	*/
	NOTIF_INFO_ACP_PROCESSING_DONE = 2,

	/**
	* Pipeline event, indicates CDI processing is finished.
	* @note This event is sent only if the Auto Exposure and Auto White Balance algorithm produces
	* new sensor settings that need to be updated in the image sensor.
	* @note Only eNotifType, uIndex & frameCaptureTSC are valid in NotificationData for this event.
	*/
	NOTIF_INFO_CDI_PROCESSING_DONE = 3,

	/**
	* Pipeline event, indicates pipeline was forced to drop a frame due to a slow consumer or system issues.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_WARN_ICP_FRAME_DROP = 100,

	/**
	* Pipeline event, indicates a discontinuity was detected in parsed embedded data frame sequence number.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_WARN_ICP_FRAME_DISCONTINUITY = 101,

	/**
	* Pipeline event, indicates occurrence of timeout while capturing.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_WARN_ICP_CAPTURE_TIMEOUT = 102,

	/**
	* Pipeline event, indicates ICP bad input stream.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_ICP_BAD_INPUT_STREAM = 200,

	/**
	* Pipeline event, indicates ICP capture failure.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_ICP_CAPTURE_FAILURE = 201,

	/**
	* Pipeline event, indicates embedded data parsing failure.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE = 202,

	/**
	* Pipeline event, indicates ISP processing failure.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_ISP_PROCESSING_FAILURE = 203,

	/**
	* Pipeline event, indicates auto control processing failure.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_ACP_PROCESSING_FAILURE = 204,

	/**
	* Pipeline event, indicates CDI set sensor control failure.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE = 205,

	/**
	* Device block event, indicates a deserializer failure.
	* @note Only eNotifType & gpioIdxs valid in NotificationData for this event.
	*/
	NOTIF_ERROR_DESERIALIZER_FAILURE = 207,

	/**
	* Device block event, indicates a serializer failure.
	* @note Only eNotifType, uLinkMask & gpioIdxs are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_SERIALIZER_FAILURE = 208,

	/**
	* Device block event, indicates a sensor failure.
	* @note Only eNotifType, uLinkMask & gpioIdxs are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_SENSOR_FAILURE = 209,

	/**
	* Pipeline event, indicates isp process failure due to recoverable errors.
	* @note Only eNotifType & uIndex are valid in NotificationData for this event.
	*/
	NOTIF_ERROR_ISP_PROCESSING_FAILURE_RECOVERABLE = 210,

	/**
	* Pipeline and device block event, indicates an unexpected internal failure.
	* @note For pipeline event, only eNotifType & uIndex are valid in NotificationData for this event.
	* @note For device block event, only eNotifType is valid in NotificationData for this event.
	*/
	NOTIF_ERROR_INTERNAL_FAILURE = 300,
};


/*
* When in orin platform, see NotificationData.
*/
typedef struct hw_video_notification_t
{
	HW_VIDEO_NOTIFTYPE							notiftype;
	u32											blockindex;
	u32											sensorindex;	// -1 means not valid
	u64											framecapturetsc;
	u64											framecapturestarttsc;
} hw_video_notification_t;

/*
* The ops is valid only when you always set i_benableinnerhandle to 1 when call getnotification function
* of hw_video_sensorpipeline_ops_t.
*/
typedef struct hw_video_sensorpipeline_notif_ops_t
{
	// reset the inner tag balwaysenableinnerhandle to 1, do not need to call it when init
	s32(*resetenableinnerhandle)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops);
	// output to *o_pissensorinerror
	s32(*issensorinerror)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, u32* o_pissensorinerror);
	// do not need to call it when init, init count is 0
	s32(*resetframedropcounter)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops);
	// output to *o_pnumframedrops
	s32(*getnumframedrops)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, u32* o_pnumframedrops);
} hw_video_sensorpipeline_notif_ops_t;

enum HW_VIDEO_NOTIFRETSTATUS
{
	HW_VIDEO_NOTIFRETSTATUS_MIN = 0,
	HW_VIDEO_NOTIFRETSTATUS_MINMINUSONE = HW_VIDEO_NOTIFRETSTATUS_MIN - 1,

	// successfully get the notification
	HW_VIDEO_NOTIFRETSTATUS_GET,
	HW_VIDEO_NOTIFRETSTATUS_TIMEOUT,
	HW_VIDEO_NOTIFRETSTATUS_QUIT,

	HW_VIDEO_NOTIFRETSTATUS_MAXADDONE,
	HW_VIDEO_NOTIFRETSTATUS_MAX = HW_VIDEO_NOTIFRETSTATUS_MAXADDONE - 1,
};

typedef struct hw_video_sensorpipeline_ops_t
{
	/*
	* 0 or 1
	* Whether the specific sensor pipeline is enabled or not.
	*/
	u32											bused;
	u32											blockindex;	// valid when bused is 1
	u32											sensorindex;	// valid when bused is 1
	struct hw_video_outputpipeline_ops_t		parrayoutput[HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_COUNT];

	void*										priv;	// internal use only

	s32(*isquit)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, u32* o_pisquit);
	/*
	* The timeout unit is us.
	* When you need to wait forever, use HW_TIMEOUT_FOREVER define, it is -1 properly.
	* When function successfully return, *o_pnotification will be set to the notification received.
	* i_buseinnerhandle is 0 or 1
	* i_buseinnerhandle is 1 meaning enable inner notification handle before return the getnotification
	* function.
	* Properly return 0, when return 0, *o_pnotifretstatus output value is valid.
	* o_pnotifretstatus should not be NULL, *o_pnotifretstatus output HW_VIDEO_NOTIFRETSTATUS.
	*/
	s32(*getnotification)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, 
		struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle, u32 i_timeoutus, 
		HW_VIDEO_NOTIFRETSTATUS* o_pnotifretstatus);
	/*
	* Get the elements count currently in the queue.
	* Output the count to *o_pcount.
	*/
	s32(*getcount)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, u32* o_pcount);

	/*
	* Valid only when you always set i_benableinnerhandle to 1 when call getnotification function of 
	* hw_video_sensorpipelinecb_t.
	*/
	struct hw_video_sensorpipeline_notif_ops_t		notif_ops;
} hw_video_sensorpipeline_ops_t;

/*
* The ops is valid only when you always set i_benableinnerhandle to 1 when call getnotification function
* of hw_video_blockpipeline_ops_t.
*/
typedef struct hw_video_blockpipeline_notif_ops_t
{
	// reset the inner tag balwaysenableinnerhandle to 1, do not need to call it when init
	s32(*resetenableinnerhandle)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops);
	// output to *o_pisblockinerror
	s32(*isblockinerror)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, u32* o_pisblockinerror);
	// output the u8* buffer pointer to *o_pperrorbuffer, output the written size to *o_psizewritten.
	s32(*getsiplerrordetails_deserializer)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, u8** o_pperrorbuffer, u32* o_psizewritten);
	// output the u8* buffer pointer to *o_pperrorbuffer, output the written size to *o_psizewritten.
	s32(*getsiplerrordetails_serializer)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, u8** o_pperrorbuffer, u32* o_psizewritten);
	// output the u8* buffer pointer to *o_pperrorbuffer, output the written size to *o_psizewritten.
	s32(*getsiplerrordetails_sensor)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, u8** o_pperrorbuffer, u32* o_psizewritten);
} hw_video_blockpipeline_notif_ops_t;

typedef struct hw_video_blockpipeline_ops_t
{
	u32											bused;	// 0 or 1
	u32											blockindex;	// valid when bused is 1
	/*
	* Valid when bused is 1.
	* Sensor number of the specific block.
	* We still need to check bused of every sensor in the parraysensor to know whether the sensor pipeline 
	* is enabled or not.
	*/
	u32											numsensors;
	struct hw_video_sensorpipeline_ops_t		parraysensor[HW_VIDEO_NUMSENSORS_PER_BLOCK];

	void*										priv;	// internal use only

	s32(*isquit)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, u32* o_pisquit);
	/*
	* The timeout unit is us.
	* When you need to wait forever, use HW_TIMEOUT_FOREVER define, it is -1 properly.
	* When function successfully return, *o_pnotification will be set to the notification received.
	* i_buseinnerhandle is 0 or 1
	* i_buseinnerhandle is 1 meaning enable inner notification handle before return the getnotification
	* function.
	* Properly return 0, when return 0, *o_pnotifretstatus output value is valid.
	* o_pnotifretstatus should not be NULL, *o_pnotifretstatus output HW_VIDEO_NOTIFRETSTATUS.
	*/
	s32(*getnotification)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, 
		struct hw_video_notification_t* o_pnotification, u32 i_benableinnerhandle, u32 i_timeoutus,
		HW_VIDEO_NOTIFRETSTATUS* o_pnotifretstatus);
	/*
	* Get the elements count currently in the queue.
	* Output the count to *o_pcount.
	*/
	s32(*getcount)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, u32* o_pcount);

	/*
	* Valid only when you always set i_benableinnerhandle to 1 when call getnotification function of
	* hw_video_blockpipeline_ops_t.
	*/
	struct hw_video_blockpipeline_notif_ops_t		notif_ops;
} hw_video_blockpipeline_ops_t;

/*
* Its name "blocks" not "block".
*/
typedef struct hw_video_blockspipeline_ops_t
{
	struct hw_video_blockpipeline_ops_t			parrayblock[HW_VIDEO_NUMBLOCKS_MAX];
} hw_video_blockspipeline_ops_t;

typedef struct hw_video_blockinfo_t
{
	u32											numsensors_byblock;
} hw_video_blockinfo_t;

typedef struct hw_video_info_t
{
	u32											numblocks;
	u32											numsensors;
	struct hw_video_blockinfo_t					parrayblockinfo[HW_VIDEO_NUMBLOCKS_MAX];

	/*
	* The pext structure is defined in the implemented module header file.
	* The memory is allocated when open function.
	* The memory should be released when close function.
	*/
	void*										pext;
} hw_video_info_t;

/*
* It is a callback function.
* The buffer information is set by hal and the user use it to do the handling operation.
* The buffer information i_pbufferinfo instance is not valid when the callback function return. 
* The i_pbufferinfo is the buffer information of the current handling frame.
*/
typedef void(*hw_video_sensorpipelinedatacb)(struct hw_video_bufferinfo_t* i_pbufferinfo);

typedef struct hw_video_sensorpipelinedatacbconfig_t
{
	// 0 or 1
	u32											bused;
	// see HW_VIDEO_REGDATACB_TYPE
	u32											type;
	hw_video_sensorpipelinedatacb				cb;
	/*
	* Valid when not CUDA type.
	* 1 mean the buffer will be release after you finish called the cb, properly you do not need 
	* to free the data buffer in the cb.
	* 0 mean the buffer will be released before you finish called the cb, properly you need to 
	* free the data buffer in the cb.
	* Currently, you'd better set bsynccb to 1.
	* It will fail if the path do not support async cb. Sync mode is definitely supported.
	*/
	u32											bsynccb;
	/*
	* custom pointer, it will transfer to the data cb function without change
	*/
	void*										pcustom;
} hw_video_sensorpipelinedatacbconfig_t;

typedef struct hw_video_sensorpipelinedatacbsconfig_t
{
	// The array number of datacbs.
	u32											arraynumdatacbs;
	/*
	* Every regdatacb type can only register valid once.
	*/
	struct hw_video_sensorpipelinedatacbconfig_t	parraydatacbs[HW_VIDEO_REGDATACB_TYPE_COUNT];
} hw_video_sensorpipelinedatacbsconfig_t;

typedef struct hw_video_sensorpipelineconfig_t
{
	// 0 or 1
	u32											bused;
	// valid when bused is 1
	u32											blockindex;
	// valid when bused is 1
	u32											sensorindex;
	// 0 or 1, valid when bused is 1
	u32											bcaptureoutputrequested;
	// 0 or 1, valid when bused is 1
	u32											bisp0outputrequested;
	// 0 or 1, valid when bused is 1
	u32											bisp1outputrequested;
	// 0 or 1, valid when bused is 1
	u32											bisp2outputrequested;
	struct hw_video_sensorpipelinedatacbsconfig_t	datacbsconfig;
} hw_video_sensorpipelineconfig_t;

typedef struct hw_video_blockpipelineconfig_t
{
	// 0 or 1
	u32											bused;
	// valid when bused is 1
	u32											blockindex;
	struct hw_video_sensorpipelineconfig_t		parraysensor[HW_VIDEO_NUMSENSORS_PER_BLOCK];
} hw_video_blockpipelineconfig_t;

/*
* One pipeline correspondent to one sensor.
* The info includes customer settings and ops
*/
typedef struct hw_video_blockspipelineconfig_t
{
	struct hw_video_blockpipelineconfig_t		parrayblock[HW_VIDEO_NUMBLOCKS_MAX];
} hw_video_blockspipelineconfig_t;

/*
* Currently, you cannot pipeline_stop and then pipeline_start without do pipeline_close and 
* pipeline_open operation.
*/
typedef struct hw_video_ops_t
{
	/*
	* You should allocate the memory of o_pinfo before call the function.
	* You can set o_pinfo to NULL to mean you do not care about the info.
	* You do not need to allocate the memory of pext of hw_video_info_t.
	*/
	s32(*device_open)(struct hw_video_t* io_pvideo, struct hw_video_info_t* o_pinfo);
	s32(*device_close)(struct hw_video_t* io_pvideo);

	/*
	* You can NOT set i_pblockspipelineconfig to NULL.
	*/
	s32(*pipeline_open)(struct hw_video_t* io_pvideo, hw_video_handlepipeline* o_phandlepipeline,
		struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig, 
		struct hw_video_blockspipeline_ops_t** o_ppblockspipeline_ops);
	s32(*pipeline_close)(struct hw_video_t* io_pvideo, hw_video_handlepipeline i_handlepipeline);
	s32(*pipeline_start)(struct hw_video_t* io_pvideo, hw_video_handlepipeline i_handlepipeline);
	/*
	* You should stop calling any notification get operation before calling pipeline_close
	* operation.
	*/
	s32(*pipeline_stop)(struct hw_video_t* io_pvideo, hw_video_handlepipeline i_handlepipeline);
} hw_video_ops_t;

typedef struct hw_video_t
{
	struct hw_device_t							common;

	struct hw_video_ops_t						ops;

	void*										priv;	// internal use only
} hw_video_t;

__END_DECLS

#endif
