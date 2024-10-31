#ifndef HW_LIDAR_MODULEID_V0_1_H
#define HW_LIDAR_MODULEID_V0_1_H

#include "hw_global.h"

#define HW_LIDAR_MODULEID_VERSION HW_MAKEV_DEVICE_MODULEID_VERSION(0, 1)

enum HW_LIDAR_MODULEID
{
    HW_LIDAR_MODULEID_MIN = 0,
    HW_LIDAR_MODULEID_MINMINUSONE = HW_LIDAR_MODULEID_MIN - 1,

    HW_LIDAR_MODULEID_SOCKET,
    HW_LIDAR_MODULEID_PCAP,

    HW_LIDAR_MODULEID_MAXADDONE,
    HW_LIDAR_MODULEID_MAX = HW_LIDAR_MODULEID_MAXADDONE - 1,
};

typedef struct hw_global_lidar_moduleid_magic_t
{
    u32 moduleid;
    u32 magic;
    const char *desc;
} hw_global_lidar_moduleid_magic_t;

static struct hw_global_lidar_moduleid_magic_t _parray_hw_lidar_devtype_magic[] = {
    {HW_LIDAR_MODULEID_SOCKET, 0x11223344, "socket"},
};
STATIC_ASSERT(sizeof(_parray_hw_lidar_devtype_magic) / sizeof(struct hw_global_lidar_moduleid_magic_t));

inline u32 hw_lidar_moduleid_magic_get(u32 i_lidar_moduleid)
{
    if (HW_UNLIKELY(i_lidar_moduleid) > HW_LIDAR_MODULEID_MAX)
    {
        return 0;
    }
    if (HW_UNLIKELY(i_lidar_moduleid != _parray_hw_lidar_devtype_magic[i_lidar_moduleid].moduleid))
    {
        return 0;
    }
    return _parray_hw_lidar_devtype_magic[i_lidar_moduleid].magic;
}

inline const char *hw_lidar_moduleid_desc_get(u32 i_lidar_moduleid)
{
    if (HW_UNLIKELY(i_lidar_moduleid) > HW_LIDAR_MODULEID_MAX)
    {
        return "NA";
    }
    if (HW_UNLIKELY(i_lidar_moduleid != _parray_hw_lidar_devtype_magic[i_lidar_moduleid].moduleid))
    {
        return "NA";
    }
    return _parray_hw_lidar_devtype_magic[i_lidar_moduleid].desc;
}

#endif // HW_LIDAR_MODULEID_V0_1_H
