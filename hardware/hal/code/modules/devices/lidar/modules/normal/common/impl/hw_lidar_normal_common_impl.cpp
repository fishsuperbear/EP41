#include "lidar/modules/normal/common/impl/hw_lidar_normal_common_impl.h"

static s32 hw_lidar_device_open(struct hw_lidar_t *io_plidar, struct hw_lidar_callback_t *i_callback)
{
    HW_LIDAR_LOG_INFO("hw_lidar_device_open\n");

    HWLidarNormalContext* pcontext = (HWLidarNormalContext*)io_plidar->priv;
    if (pcontext == nullptr)
    {
        HW_LIDAR_LOG_ERR("HWLidarNormalContext is nullptr!\n");
        return -1;
    }
    pcontext->Device_Open(i_callback);

	return 0;
}

static s32 hw_lidar_device_close(struct hw_lidar_t *io_plidar)
{
    HW_LIDAR_LOG_INFO("hw_lidar_device_close\n");
    HWLidarNormalContext* pcontext = (HWLidarNormalContext*)io_plidar->priv;
    if (pcontext == nullptr)
    {
        HW_LIDAR_LOG_ERR("HWLidarNormalContext is nullptr!\n");
        return -1;
    }
    pcontext->Device_Close();

	return 0;
}

s32 hw_lidar_setlidarops(struct hw_lidar_t* io_plidar)
{
    io_plidar->ops.device_read_open = hw_lidar_device_open;
    io_plidar->ops.device_read_close = hw_lidar_device_close;
    return 0;
}