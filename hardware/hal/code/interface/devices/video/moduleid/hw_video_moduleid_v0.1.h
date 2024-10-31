#ifndef HW_VIDEO_MODULEID_V0_1_H
#define HW_VIDEO_MODULEID_V0_1_H

#include "hw_global.h"

#define HW_DEVICE_MODULEID_VERSION HW_MAKEV_DEVICE_MODULEID_VERSION(0, 1)

/*
 * Different module may implement the same device api.
 * The same module may support different version device api.
 */
enum HW_VIDEO_MODULEID {
    HW_VIDEO_MODULEID_MIN = 0,
    HW_VIDEO_MODULEID_MINMINUSONE = HW_VIDEO_MODULEID_MIN - 1,

    HW_VIDEO_MODULEID_NVMEDIA_IMX728,
    HW_VIDEO_MODULEID_NVMEDIA_GROUPA,
    HW_VIDEO_MODULEID_NVMEDIA_GROUPB,
    HW_VIDEO_MODULEID_NVMEDIA_GROUPC,
    HW_VIDEO_MODULEID_NVMEDIA_MULTIGROUP,
    HW_VIDEO_MODULEID_NVMEDIA_IPC_PRODUCER,
    HW_VIDEO_MODULEID_NVMEDIA_IPC_CONSUMER_CUDA,
    HW_VIDEO_MODULEID_NVMEDIA_IPC_CONSUMER_ENC,
    HW_VIDEO_MODULEID_V4L2,

    HW_VIDEO_MODULEID_MAXADDONE,
    HW_VIDEO_MODULEID_MAX = HW_VIDEO_MODULEID_MAXADDONE - 1,
};

typedef struct hw_global_video_moduleid_magic_t {
    u32 moduleid;
    u32 magic;
    const char* desc;
} hw_global_video_moduleid_magic_t;

static struct hw_global_video_moduleid_magic_t _parray_hw_video_moduleid_magic[] =
    {
        {HW_VIDEO_MODULEID_NVMEDIA_IMX728, 0x4A2F5B3D, "nvmedia_imx728"},
        {HW_VIDEO_MODULEID_NVMEDIA_GROUPA, 0x6C2D576A, "nvmedia_groupa"},
        {HW_VIDEO_MODULEID_NVMEDIA_GROUPB, 0x7D3A48FE, "nvmedia_groupb"},
        {HW_VIDEO_MODULEID_NVMEDIA_GROUPC, 0x8A4D59DE, "nvmedia_groupc"},
        {HW_VIDEO_MODULEID_NVMEDIA_MULTIGROUP, 0x9B3C275E, "nvmedia_multigroup"},
        {HW_VIDEO_MODULEID_NVMEDIA_IPC_PRODUCER, 0x7F673A98, "nvmedia_ipc_producer"},
        {HW_VIDEO_MODULEID_NVMEDIA_IPC_CONSUMER_CUDA, 0x3F5A4F4B, "nvmedia_ipc_consumer_cuda"},
        {HW_VIDEO_MODULEID_NVMEDIA_IPC_CONSUMER_ENC, 0x3F5A4F4C, "nvmedia_ipc_consumer_enc"},
};
STATIC_ASSERT(sizeof(_parray_hw_video_moduleid_magic) / sizeof(struct hw_global_video_moduleid_magic_t));

inline u32 hw_video_moduleid_magic_get(u32 i_video_moduleid) {
    if (HW_UNLIKELY(i_video_moduleid > HW_VIDEO_MODULEID_MAX)) return 0;
    if (HW_UNLIKELY(i_video_moduleid != _parray_hw_video_moduleid_magic[i_video_moduleid].moduleid)) return 0;
    return _parray_hw_video_moduleid_magic[i_video_moduleid].magic;
}
inline const char* hw_video_moduleid_desc_get(u32 i_video_moduleid) {
    if (HW_UNLIKELY(i_video_moduleid > HW_VIDEO_MODULEID_MAX)) return "NA";
    if (HW_UNLIKELY(i_video_moduleid != _parray_hw_video_moduleid_magic[i_video_moduleid].moduleid)) return "NA";
    return _parray_hw_video_moduleid_magic[i_video_moduleid].desc;
}

#endif
