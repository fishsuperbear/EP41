#ifndef HW_NVMEDIA_COMMON_H
#define HW_NVMEDIA_COMMON_H

#include "hw_nvmedia_def.h"

typedef struct hw_nvmedia_module_t {
	/*
	* The description const string of drive os version. Like 6.0.4
	*/
	const char*			driveosversiondesc;
} hw_nvmedia_module_t;

s32 hw_nvmedia_setvideoops(struct hw_video_t* io_pvideo);

#define ALIGN_STEP 512
#define ALIGN(width) ((width+ALIGN_STEP-1)&(~(ALIGN_STEP-1)))

typedef struct hw_nvmedia_cuda_datainfo_t {
	int width;
	int height;
	int stride;
} hw_nvmedia_cuda_datainfo_t;

#endif
