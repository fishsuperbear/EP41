#include "lidar/modules/normal/hw_lidar_normal.h"

extern "C" struct hw_lidar_module_t HAL_MODULE_INFO_SYM;

struct hw_lidar_module_t HAL_MODULE_INFO_SYM = {
    .common = {
        .tag = HARDWARE_MODULE_TAG,
        .hal_api_version = HARDWARE_HAL_API_VERSION,
        .global_devicetype_version = HW_GLOBAL_DEVTYPE_VERSION,
        .devicetype = HW_DEVICETYPE_LIDAR,
        .module_id = HW_LIDAR_MODULEID_SOCKET,
        .devtype_magic = hw_global_devtype_magic_get(HW_DEVICETYPE_LIDAR),
        .device_moduleid_version = HW_LIDAR_MODULEID_VERSION,
        .module_api_version = HW_LIDAR_MODULE_API_VERSION,
        .devmoduleid_magic = hw_lidar_moduleid_magic_get(HW_LIDAR_MODULEID_SOCKET),
        .description = "lidar demo",
        .privdata = {
            .dso = NULL,
            .pvoid = NULL,
        },
        .privapi = {
            .init = hw_module_privapi_init,
            .check_device_api_version = hw_module_privapi_check_device_api_version,
            .trigger_check_unmasklogleft = hw_module_privapi_trigger_check_unmasklogleft,
            .device_get = hw_module_privapi_device_get,
            .device_put = hw_module_privapi_device_put,
        },
    },
};

static LidarNormalEnv env_;

s32 hw_module_privapi_init(void **io_ppvoid)
{
    HW_LIDAR_LOG_INFO("hw_module_privapi_init\n");
    return 0;
}

s32 hw_module_privapi_check_device_api_version(u32 i_device_api_version)
{
    HW_LIDAR_LOG_INFO("hw_module_privapi_check_device_api_version\n");
    return 0;
}

s32 hw_module_privapi_trigger_check_unmasklogleft()
{
    HW_LIDAR_LOG_INFO("hw_module_privapi_trigger_check_unmasklogleft\n");
    return 0;
}

s32 hw_module_privapi_device_get(struct hw_module_t *i_pmodule, void *i_param, struct hw_device_t **io_ppdevice)
{
    HW_LIDAR_LOG_INFO("hw_module_privapi_device_get\n");

    s32 ret;
    struct hw_lidar_t *plidar = (struct hw_lidar_t *)malloc(sizeof(struct hw_lidar_t));
    plidar->common.tag = HARDWARE_DEVICE_TAG;
    plidar->common.device_api_version = env_.device_api_version_;
    plidar->common.pmodule = (hw_module_t *)&HAL_MODULE_INFO_SYM;
    hw_lidar_setlidarops(plidar);
    *io_ppdevice = (struct hw_device_t *)plidar;

    if (env_.pcontext == nullptr)
    {
        env_.pcontext = new HWLidarNormalContext(plidar);
        ret = env_.pcontext->Init();
        if (ret != 0)
        {
            HW_LIDAR_LOG_ERR("HWLidarNormalContext init failed!\n");
            return -2;
        }
    }
    plidar->priv = env_.pcontext;

    return 0;
}

s32 hw_module_privapi_device_put(struct hw_module_t *i_pmodule, struct hw_device_t *i_pdevice)
{
    HW_LIDAR_LOG_INFO("hw_module_privapi_device_put\n");

    struct hw_lidar_t* plidar = (struct hw_lidar_t*)i_pdevice;
    HWLidarNormalContext *pcontext = (HWLidarNormalContext *)plidar->priv;
    delete(pcontext);
    delete(plidar);

    return 0;
}

