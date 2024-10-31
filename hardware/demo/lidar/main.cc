#include <stdio.h>
#include <thread>
#include "hardware.h"

#define LIDAR_MODULE_NAME "../../../hal/code/lib/libhw_lidar_vs.so"

void onLidarDataCallback(int index, struct hw_lidar_pointcloud_XYZIT *points, int size)
{
    printf("receive lidar%d pointcloud, size: %d, timestamp: %lld\n", 
           index, 
           size, 
           points[size - 1].timestamp);
}

int main(int argc, char** argv)
{
    u32 ret;
    struct hw_module_t *pmodule;
    struct hw_lidar_t *plidar;

    /*
    * Register default sig handler.
    */
    hw_plat_regsighandler_default();

    ret = hw_module_get(LIDAR_MODULE_NAME, &pmodule);
    if (ret < 0)
    {
        printf("hw_module_get failed!\n");
        return ret;
    }
    
    ret = hw_module_device_get(pmodule, NULL, (hw_device_t**)&plidar);
    if (ret < 0)
    {
        printf("hw_module_device_get failed!\n");
        return ret;
    }

    int lidar_number = 2;
    hw_lidar_config_info_t config[lidar_number];
    // lidar1
    config[0].index = 1;
    config[0].lidar_model = HW_LIDAR_MODEL::ROBOSENSE_M1;
    config[0].port = 6699;
    // lidar2
    config[1].index = 2;
    config[1].lidar_model = HW_LIDAR_MODEL::ROBOSENSE_M1;
    config[1].port = 6670;

    struct hw_lidar_callback_t callback;
    callback.configs = config;
    callback.config_number = lidar_number;
    callback.data_cb = onLidarDataCallback;

    for (int i = 0; i < callback.config_number; i++)
    {
        printf("lidar%d config, model: %d, port: %d\n", 
               callback.configs[i].index, 
               callback.configs[i].lidar_model,
               callback.configs[i].port);
    }

    struct hw_lidar_info_t lidarinfo;
    std::thread thread_open([&]() 
    {
        int ret = plidar->ops.device_read_open(plidar, &callback);
    });
    thread_open.detach();

    sleep(1000);
    ret = plidar->ops.device_read_close(plidar);

    return 0;
}