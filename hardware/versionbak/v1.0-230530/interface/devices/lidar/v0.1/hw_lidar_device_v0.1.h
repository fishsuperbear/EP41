#ifndef HW_LIDAR_DEVICE_V0_1_H
#define HW_LIDAR_DEVICE_V0_1_H

#include "hw_video_version.h"

__BEGIN_DECLS

struct hw_lidar_t;

enum LIDAR_ERROR_CODE
{
    SUCCESS = 0,
    UNKNOWN_LIDAR_MODEL = 1,
};

typedef struct hw_lidar_pointcloud_XYZIT
{
    float x;
    float y;
    float z;
    u32 intensity; // 0-255
    s64 timestamp; // ns
} hw_lidar_pointcloud_XYZIT;

typedef struct hw_lidar_info_t
{

} hw_lidar_info_t;

enum HW_LIDAR_MODEL
{
    ROBOSENSE_M1 = 1,
};

typedef struct hw_lidar_config_info_t
{
    int index;
    HW_LIDAR_MODEL lidar_model;
    int port;
} hw_lidar_config_info_t;

typedef void(*hw_lidar_pointcloud_callback)(int index, struct hw_lidar_pointcloud_XYZIT *points, int size);

typedef struct hw_lidar_callback_t
{
    hw_lidar_config_info_t *configs;
    int config_number;
    hw_lidar_pointcloud_callback data_cb;
} hw_lidar_callback_t;

typedef struct hw_lidar_ops_t
{
    s32 (*device_read_open)(struct hw_lidar_t *io_pvideo, struct hw_lidar_callback_t *i_callback);
    s32 (*device_read_close)(struct hw_lidar_t *io_pvideo);
} hw_lidar_ops_t;

typedef struct hw_lidar_t
{
    struct hw_device_t common;
    struct hw_lidar_ops_t ops;
    void *priv;
} hw_lidar_t;

__END_DECLS

#endif // HW_LIDAR_DEVICE_V0_1_H