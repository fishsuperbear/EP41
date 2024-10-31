#ifndef HAL_CAMERA_DEF_IMPL_H
#define HAL_CAMERA_DEF_IMPL_H

#include "hal_camera_baseinc_impl.h"

/*
* You may not care about it.
* BE CAREFUL! The following function usage may change when hal camera version upgrade.
* The input i_outputtype see HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE defines.
* The query info of the camera device sensor info of specific opentype and specific block type.
* CameraDeviceOpenTypeSensorInfo may be inherited in the later version.
*/
typedef void (*camera_device_custom_threadroutine_handleframe_impl)(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops,
    void* i_pcontext, u32 i_outputtype);

/*
* You may not care about it.
* BE CAREFUL! The following function usage may change when hal camera version upgrade.
* The query info of the camera device block info of specific opentype.
* CameraDeviceOpenTypeBlockInfo may be inherited in the later version.
*/
typedef void (*camera_device_custom_threadroutine_sensornotif_impl)(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops,
    void* i_pcontext);

/*
* You may not care about it.
* BE CAREFUL! The following function usage may change when hal camera version upgrade.
* CameraDeviceDataCbRegInfo may be inherited in the later version.
*/
typedef void (*camera_device_custom_threadroutine_blocknotif_impl)(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops,
    void* i_pcontext);

#endif
