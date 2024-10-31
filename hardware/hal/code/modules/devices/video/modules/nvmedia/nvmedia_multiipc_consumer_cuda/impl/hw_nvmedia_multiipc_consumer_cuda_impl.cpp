#include "hw_nvmedia_multiipc_consumer_cuda_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"

HWNvmediaMultiIpcConsumerCudaContext::HWNvmediaMultiIpcConsumerCudaContext(struct hw_video_t* i_pvideo)
	: HWNvmediaIpcConsumerContext(i_pvideo)
{
	/*
	* 1 mean use mailbox mode.
	*/
#ifdef BOARD_CONFIG_NVIDIA
	_deviceopenpara.Init(HW_NVMEDIA_APPTYPE_IPC_CONSUMER_CUDA,
                             "HZ_multi_camera",
                             "0x0010 0x0001 0x1111", 1, "/usr/share/camera/");
#else
	_deviceopenpara.Init(HW_NVMEDIA_APPTYPE_IPC_CONSUMER_CUDA,
                             "XPC_MULTIPLE_CAMERA",
                             "0x1011 0x1111 0x1111 0", 1, "/usr/share/camera/");
#endif
		/* "HZ_multi_camera", */
        /* "0x0010 0x1101 0x1111", 1, "/usr/share/camera/"); */
                             //"V1SIM728S1RU3120NB20_CPHY_x4",
                             //"0x1000 0 0 0", 1, "/usr/share/camera/");
                             //
                             //
		//"HZ_multi_camera",
		//"0x0010 0x1111 0x1111", 1, "/usr/share/camera/");
}

hw_ret_s32 HWNvmediaMultiIpcConsumerCudaContext::GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype,
	HW_VIDEO_INPUT_TYPE inputType, HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig)
{
	o_pcbconfig->bdirectcb = 1;
	switch (inputType)
    {
        case HW_VIDEO_INPUT_TYPE_RAW12:
            switch (i_regdatacbtype)
            {
                case HW_VIDEO_REGDATACB_TYPE_RAW12:
                    // when imx728, you need icp to output rgb
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
                    o_pcbconfig->consumertype = COMMON_CONSUMER;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_RAW12;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_YUV420:
                    // when imx728, you need isp0 to output yuv420
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0;
                    o_pcbconfig->consumertype = COMMON_CONSUMER;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PRIV;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 1;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_YUV420_PRIV:
                    // when imx728, you need isp0 to output yuv420
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0;
                    o_pcbconfig->consumertype = COMMON_CONSUMER;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PRIV;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_CUDA:
                    o_pcbconfig->consumertype = CUDA_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_RGBA;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_AVC:
                    o_pcbconfig->consumertype = ENC_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_AVC;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_HEVC:
                    o_pcbconfig->consumertype = ENC_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ISP0;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_HEVC;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                default:
                    HW_NVMEDIA_LOG_ERR("No correspondent output type of input regdatacbtype[%u] failed.\r\n", i_regdatacbtype);
                    return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_NO_CORRESPONDENT_OUTPUTTYPE);
            }

        case HW_VIDEO_INPUT_TYPE_YUV422:
            switch (i_regdatacbtype)
            {
                case HW_VIDEO_REGDATACB_TYPE_YUV422:
                    // when isx021, you need icp to output yuv422
                    o_pcbconfig->consumertype = COMMON_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV422;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_YUV420:
                    // when isx021, you need vic to output yuv420 bl and use get pixel function to output yuv420 pl
                    o_pcbconfig->consumertype = VIC_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PRIV;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 1;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_YUV420_PRIV:
                    // when isx021, you need vic to output yuv420 bl
                    o_pcbconfig->consumertype = VIC_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PRIV;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_AVC:
                    o_pcbconfig->consumertype = ENC_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_AVC;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_HEVC:
                    o_pcbconfig->consumertype = ENC_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_HEVC;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                case HW_VIDEO_REGDATACB_TYPE_CUDA:
                    o_pcbconfig->consumertype = CUDA_CONSUMER;
                    o_pcbconfig->outputtype = HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_ICP;
                    o_pcbconfig->commonconsumerconfig.expectedoriginsubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PRIV;
                    o_pcbconfig->commonconsumerconfig.bneedgetpixel = 0;
                    return 0;
                default:
                    HW_NVMEDIA_LOG_ERR("No correspondent output type of input regdatacbtype[%u] failed.\r\n", i_regdatacbtype);
                    return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_NO_CORRESPONDENT_OUTPUTTYPE);
            }
        default:
            HW_NVMEDIA_LOG_ERR("No correspondent output type of input regdatacbtype[%u] failed.\r\n", i_regdatacbtype);
            return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_NO_CORRESPONDENT_OUTPUTTYPE);
    }
}
