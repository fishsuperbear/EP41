#include "hw_nvmedia_common_impl.h"

static s32 hw_nvmedia_device_open(struct hw_video_t* io_pvideo, struct hw_video_info_t* o_pinfo)
{
	HWNvmediaContext* pnvmedia = (HWNvmediaContext*)io_pvideo->priv;
	return pnvmedia->Device_Open(o_pinfo);
}

static s32 hw_nvmedia_device_close(struct hw_video_t* io_pvideo)
{
	HWNvmediaContext* pnvmedia = (HWNvmediaContext*)io_pvideo->priv;
	return pnvmedia->Device_Close();
}

static s32 hw_nvmedia_pipeline_open(struct hw_video_t* io_pvideo, hw_video_handlepipeline* o_phandlepipeline,
	struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig,
	struct hw_video_blockspipeline_ops_t** o_ppblockspipeline_ops)
{
	HWNvmediaContext* pnvmedia = (HWNvmediaContext*)io_pvideo->priv;
	return pnvmedia->Pipeline_Open(i_pblockspipelineconfig, o_ppblockspipeline_ops);
}

static s32 hw_nvmedia_pipeline_close(struct hw_video_t* io_pvideo, hw_video_handlepipeline i_handlepipeline)
{
	HWNvmediaContext* pnvmedia = (HWNvmediaContext*)io_pvideo->priv;
	return pnvmedia->Pipeline_Close();
}

static s32 hw_nvmedia_pipeline_start(struct hw_video_t* io_pvideo, hw_video_handlepipeline i_handlepipeline)
{
	HWNvmediaContext* pnvmedia = (HWNvmediaContext*)io_pvideo->priv;
	return pnvmedia->Pipeline_Start();
}

static s32 hw_nvmedia_pipeline_stop(struct hw_video_t* io_pvideo, hw_video_handlepipeline i_handlepipeline)
{
	HWNvmediaContext* pnvmedia = (HWNvmediaContext*)io_pvideo->priv;
	return pnvmedia->Pipeline_Stop();
}

s32 hw_nvmedia_setvideoops(struct hw_video_t* io_pvideo)
{
	io_pvideo->ops.device_open = hw_nvmedia_device_open;
	io_pvideo->ops.device_close = hw_nvmedia_device_close;
	io_pvideo->ops.pipeline_open = hw_nvmedia_pipeline_open;
	io_pvideo->ops.pipeline_close = hw_nvmedia_pipeline_close;
	io_pvideo->ops.pipeline_start = hw_nvmedia_pipeline_start;
	io_pvideo->ops.pipeline_stop = hw_nvmedia_pipeline_stop;
	return 0;
}
