#include "hw_nvmedia_groupc_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"

HWNvmediaGroupcContext::HWNvmediaGroupcContext(struct hw_video_t* i_pvideo)
	: HWNvmediaSingleProcessContext(i_pvideo)
{
#if 1
	/*
	* 1 mean use mailbox mode.
	*/
	_deviceopenpara.Init(HW_NVMEDIA_APPTYPE_SINGLE_PROCESS,
		"OVX1F_DPHY_x4",
		"0x0000 0x0000 0x1111 0x0000", 1, "/opt/nvidia/nvmedia/nit/");
#endif
}

hw_ret_s32 HWNvmediaGroupcContext::GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype,
	HW_VIDEO_INPUT_TYPE inputType, HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig)
{
	o_pcbconfig->bdirectcb = 1;
	switch (i_regdatacbtype)
	{
	case HW_VIDEO_REGDATACB_TYPE_YUV422:
		// when ovx1f, you need icp to output yuv422
		o_pcbconfig->consumertype = COMMON_CONSUMER;
		o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
		o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV422;
		o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
		return 0;
	default:
		HW_NVMEDIA_LOG_ERR("No correspondent output type of input regdatacbtype[%u] failed.\r\n", i_regdatacbtype);
		return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_NO_CORRESPONDENT_OUTPUTTYPE);
	}
}
