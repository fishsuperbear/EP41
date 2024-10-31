#include "hw_nvmedia_imx728_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"

HWNvmediaImx728Context::HWNvmediaImx728Context(struct hw_video_t* i_pvideo)
	: HWNvmediaSingleProcessContext(i_pvideo)
{
#if 1
	/*
	* 1 mean use mailbox mode.
	*/
	_deviceopenpara.Init(HW_NVMEDIA_APPTYPE_SINGLE_PROCESS,
		"V1SIM728S1RU3120NB20_CPHY_x4",
		"0x1000 0 0 0", 1, "/opt/nvidia/nvmedia/nit/");
#endif
#if 0
	/*
	* 1 mean use mailbox mode.
	*/
	_deviceopenpara.Init(HW_NVMEDIA_APPTYPE_SINGLE_PROCESS,
		"ISX021_DPHY_x4",
		"0 0x0001 0 0", 1, "/opt/nvidia/nvmedia/nit/");
#endif
#if 0
	/*
	* 1 mean use mailbox mode.
	*/
	_deviceopenpara.Init(HW_NVMEDIA_APPTYPE_SINGLE_PROCESS,
		"OVX1F_DPHY_x4",
		"0 0 0x1111 0", 1, "/opt/nvidia/nvmedia/nit/");
#endif
}

hw_ret_s32 HWNvmediaImx728Context::GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype, 
	HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig)
{
	o_pcbconfig->bdirectcb = 1;
	switch (i_regdatacbtype)
	{
	case HW_VIDEO_REGDATACB_TYPE_YUV420:
		// when imx728, you need isp0 to output yuv420
		o_pcbconfig->consumertype = COMMON_CONSUMER;
		o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0;
		return 0;
	case HW_VIDEO_REGDATACB_TYPE_RGBA:
		// when imx728, you need icp to output rgb
		o_pcbconfig->consumertype = COMMON_CONSUMER;
		o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
		return 0;
	default:
		HW_NVMEDIA_LOG_ERR("No correspondent output type of input regdatacbtype[%u] failed.\r\n", i_regdatacbtype);
		return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_NO_CORRESPONDENT_OUTPUTTYPE);
	}
}
