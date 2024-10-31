#include "hal_camera_impl.h"

void Internal_ThreadRoutine_SensorPipeline_Default(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops, void* i_pcontext)
{
    s32 ret;
    struct hw_video_notification_t blocknotif;
    HW_VIDEO_NOTIFRETSTATUS notifretstatus;
    while (1)
    {
        ret = i_psensorpipeline_ops->getnotification(i_psensorpipeline_ops, &blocknotif, 1, HW_TIMEOUT_US_DEFAULT, &notifretstatus);
        if (ret == 0)
        {
            // receive notification, handle it
            if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_QUIT)
            {
                HAL_CAMERA_LOG_UNMASK("Sensor notif thread quit by restatus[quit]...\r\n");
                break;
            }
            else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_TIMEOUT)
            {
                HAL_CAMERA_LOG_DEBUG("Sensor notif receive timeout\r\n");
            }
            else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_GET)
            {
                HAL_CAMERA_LOG_UNMASK("[%d][%d]Sensor notif: %u\r\n", i_psensorpipeline_ops->blockindex,i_psensorpipeline_ops->sensorindex,blocknotif.notiftype);
            }
            else
            {
                HAL_CAMERA_LOG_ERR("Sensor Unexpected notifretstatus value[%u]\r\n", notifretstatus);
            }
        }
        else
        {
            HAL_CAMERA_LOG_ERR("Sensor Unexpected ret value[0x%x]\r\n", ret);
        }
    }
}
