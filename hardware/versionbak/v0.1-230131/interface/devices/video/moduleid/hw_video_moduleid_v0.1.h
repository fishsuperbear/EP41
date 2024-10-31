#ifndef HW_VIDEO_MODULEID_V0_1_H
#define HW_VIDEO_MODULEID_V0_1_H

#include "hw_global.h"

#define HW_DEVICE_MODULEID_VERSION		HW_MAKEV_DEVICE_MODULEID_VERSION(0,1)

/*
* Different module may implement the same device api.
* The same module may support different version device api.
*/
enum HW_VIDEO_MODULEID
{
	HW_VIDEO_MODULEID_MIN = 0,
	HW_VIDEO_MODULEID_MINMINUSONE = HW_VIDEO_MODULEID_MIN - 1,

	HW_VIDEO_MODULEID_NVMEDIA,
	HW_VIDEO_MODULEID_V4L2,

	HW_VIDEO_MODULEID_MAXADDONE,
	HW_VIDEO_MODULEID_MAX = HW_VIDEO_MODULEID_MAXADDONE - 1,
};

typedef struct hw_global_video_moduleid_magic_t
{
	u32				moduleid;
	u32				magic;
	const char*		desc;
} hw_global_video_moduleid_magic_t;

static struct hw_global_video_moduleid_magic_t _parray_hw_video_moduleid_magic[] =
{
	{HW_VIDEO_MODULEID_NVMEDIA, 0x4A2F5B3D, "nvmedia"},

};
STATIC_ASSERT(sizeof(_parray_hw_video_moduleid_magic) / sizeof(struct hw_global_video_moduleid_magic_t));

inline u32 hw_video_moduleid_magic_get(u32 i_video_moduleid)
{
	if (HW_UNLIKELY(i_video_moduleid > HW_VIDEO_MODULEID_MAX)) return 0;
	if (HW_UNLIKELY(i_video_moduleid != _parray_hw_video_moduleid_magic[i_video_moduleid].moduleid)) return 0;
	return _parray_hw_video_moduleid_magic[i_video_moduleid].magic;
}
inline const char* hw_video_moduleid_desc_get(u32 i_video_moduleid)
{
	if (HW_UNLIKELY(i_video_moduleid > HW_VIDEO_MODULEID_MAX)) return "NA";
	if (HW_UNLIKELY(i_video_moduleid != _parray_hw_video_moduleid_magic[i_video_moduleid].moduleid)) return "NA";
	return _parray_hw_video_moduleid_magic[i_video_moduleid].desc;
}

#endif
