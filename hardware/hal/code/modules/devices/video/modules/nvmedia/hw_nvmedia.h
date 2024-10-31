#ifndef HW_NVMEDIA_H
#define HW_NVMEDIA_H

#include "hw_nvmedia_compile.h"
#include "hw_nvmedia_baseinc.h"
#include "hw_nvmedia_def.h"
#include "hw_nvmedia_common.h"
#if (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_IMX728)
#include "hw_nvmedia_imx728.h"
#elif (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_GROUPA)
#include "hw_nvmedia_groupa.h"
#elif (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_GROUPB)
#include "hw_nvmedia_groupb.h"
#elif (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_MULTIGROUP)
#include "hw_nvmedia_multigroup.h"
#elif (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_IPC_CONSUMER_CUDA)
#include "hw_nvmedia_ipc_consumer_cuda.h"
#endif

#endif
