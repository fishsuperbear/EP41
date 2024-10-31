#ifndef HW_TAG_AND_VERSION_H
#define HW_TAG_AND_VERSION_H

#include "hw_platform.h"

/*
* Value for the hw_module_t/hw_device_t tag field
*/
#define MAKE_TAG_CONSTANT(A,B,C,D)							(((A) << 24) | ((B) << 16) | ((C) << 8) | (D))

#define HARDWARE_MODULE_TAG									MAKE_TAG_CONSTANT('H', 'W', 'M', 'T')
#define HARDWARE_DEVICE_TAG									MAKE_TAG_CONSTANT('H', 'W', 'D', 'T')

#define HARDWARE_MAKE_VERSION(maj,min)						((((maj) & 0xff) << 8) | ((min) & 0xff))

/*
* return 1 means va and vb are the same major version
*/
#define HW_CHECK_MAJ_VERSION(va, vb)						(((va) & 0xff00) == ((vb) & 0xff00))

/*
* Make version of hal_api_version of hw_module_t.
*/
#define HW_MAKEV_HAL_API_VERSION(maj,min)					HARDWARE_MAKE_VERSION(maj,min)
/*
* It will not change often.
*/
#define HARDWARE_HAL_API_VERSION							HW_MAKEV_HAL_API_VERSION(0,0)

/*
* Make version of global_devicetype_version of hw_module_t.
*/
#define HW_MAKEV_GLOBAL_DEVICETYPE_VERSION(maj,min)			HARDWARE_MAKE_VERSION(maj,min)

/*
* Make version of device_moduleid_version of hw_module_t.
*/
#define HW_MAKEV_DEVICE_MODULEID_VERSION(maj,min)			HARDWARE_MAKE_VERSION(maj,min)

/*
* Make version of module_api_version of hw_module_t.
*/
#define HW_MAKEV_MODULE_API_VERSION(maj,min)				HARDWARE_MAKE_VERSION(maj,min)

/*
* Make version of device_api_version of hw_device_t.
*/
#define HW_MAKEV_DEVICE_API_VERSION(maj,min)				HARDWARE_MAKE_VERSION(maj,min)

/*
* https://hozonauto.feishu.cn/wiki/RVvVwIOXPiQ3e2koQTqcGFaNnn1
*/
#define HW_NVMEDIA_VERSION "1.0.1"

#endif
