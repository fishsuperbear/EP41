#ifndef HW_GLOBAL_DEVTYPE_V0_1_H
#define HW_GLOBAL_DEVTYPE_V0_1_H

#include "hw_hal_api.h"

#define HW_GLOBAL_DEVTYPE_VERSION		HW_MAKEV_GLOBAL_DEVICETYPE_VERSION(0,1)

/*
* Different module may use the same device structure, we use the device type to tell the
* difference between modules which use different device structure(hw_device_t).
*/
enum HW_DEVICETYPE
{
	HW_DEVICETYPE_MIN = 0,
	HW_DEVICETYPE_MINMINUSONE = HW_DEVICETYPE_MIN - 1,

	HW_DEVICETYPE_VIDEO,

	HW_DEVICETYPE_MAXADDONE,
	HW_DEVICETYPE_MAX = HW_DEVICETYPE_MAXADDONE - 1,
};

typedef struct hw_global_devtype_magic_t
{
	u32				devtype;
	u32				magic;
	const char*		desc;
} hw_global_devtype_magic_t;

static struct hw_global_devtype_magic_t _parray_hw_global_devtype_magic[] =
{
	{HW_DEVICETYPE_VIDEO, 0x5E7A8C2D, "video"},

};
STATIC_ASSERT(sizeof(_parray_hw_global_devtype_magic) / sizeof(struct hw_global_devtype_magic_t) == HW_DEVICETYPE_MAXADDONE);

inline u32 hw_global_devtype_magic_get(u32 i_devtype)
{
	if (HW_UNLIKELY(i_devtype > HW_DEVICETYPE_MAX)) return 0;
	if (HW_UNLIKELY(i_devtype != _parray_hw_global_devtype_magic[i_devtype].devtype)) return 0;
	return _parray_hw_global_devtype_magic[i_devtype].magic;
}
inline const char* hw_global_devtype_desc_get(u32 i_devtype)
{
	if (HW_UNLIKELY(i_devtype > HW_DEVICETYPE_MAX)) return "NA";
	if (HW_UNLIKELY(i_devtype != _parray_hw_global_devtype_magic[i_devtype].devtype)) return "NA";
	return _parray_hw_global_devtype_magic[i_devtype].desc;
}

#endif
