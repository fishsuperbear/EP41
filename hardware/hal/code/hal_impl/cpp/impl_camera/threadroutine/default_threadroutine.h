#include "hal_camera_log_impl.h"

void Internal_ThreadRoutine_OutputPipeline_Default(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops, void* i_pcontext,
    u32 i_outputtype);
void Internal_ThreadRoutine_SensorPipeline_Default(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, void* i_pcontext);
void Internal_ThreadRoutine_BlockPipeline_Default(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops, void* i_pcontext);
