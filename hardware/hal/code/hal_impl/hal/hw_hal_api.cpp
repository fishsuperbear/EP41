#include "hw_hal_api.h"
/*
* We need to add the total head file so that we can know the device correspondent version.
*/
#include "hardware.h"

#include "hw_hal_impl_log.h"

s32 hw_module_get(const char* i_modulesoname, struct hw_module_t** o_ppmodule)
{
	/*
	* Let the os system to count the reference, we do not record it ourselves.
	*/
	struct hw_module_t* hmi = NULL;
	const char* sym = HAL_MODULE_INFO_SYM_AS_STR;
	void* handle = NULL;
	handle = dlopen(i_modulesoname, RTLD_NOW);
    if (!handle) {
        HW_HAL_LOG_ERR("dlopen error: %s", dlerror());
        return -1;
	}
	hmi = (struct hw_module_t*)dlsym(handle, sym);
	void* pvoid;
	s32 ret;
	if (hmi->privapi.init(&pvoid) == 0)
	{
		HW_HAL_LOG_UNMASK("so name[%s] init success!\r\n", i_modulesoname);
		HW_HAL_LOG_UNMASK("so details: devtype[%s]moduleid[%s]\r\n", hw_global_devtype_desc_get(hmi->devicetype), hw_video_moduleid_desc_get(hmi->module_id));
		hmi->privdata.dso = handle;
		hmi->privdata.pvoid = pvoid;
		/*
		* When we first init the module, we need to tell the module the current device version 
		* by calling check_device_api_version function.
		*/
		u32 device_api_version;
		/*
		* Input the correspondent device api version according to the module device type.
		*/
		switch (hmi->devicetype)
		{
		case HW_DEVICETYPE_VIDEO:
			device_api_version = HW_VIDEO_API_VERSION;
			if (hw_video_moduleid_magic_get(hmi->module_id) != hmi->devmoduleid_magic)
			{
				HW_HAL_LOG_UNMASK("check module_id (of the specific devicetype) magic fail!\r\n");
				return -1;
			}
			break;
		case HW_DEVICETYPE_LIDAR:
			device_api_version = HW_LIDAR_API_VERSION;
			if (hw_lidar_moduleid_magic_get(hmi->module_id) != hmi->devmoduleid_magic)
			{
				HW_HAL_LOG_UNMASK("check module_id (of the specific devicetype) magic fail!\r\n");
				return -1;
			}
			break;
		default:
			HW_HAL_LOG_UNMASK("unexpected devicetype[%d]!\r\n", hmi->devicetype);
			return -1;
		}
		ret = hmi->privapi.check_device_api_version(device_api_version);
		if (ret < 0)
		{
			HW_HAL_LOG_UNMASK("so name[%s] check_device_api_version fail!\r\n", i_modulesoname);
			return -1;
		}
		/*
		* Check tag and version one by one.
		*/
		if (hmi->tag != HARDWARE_MODULE_TAG)
		{
			HW_HAL_LOG_UNMASK("check tag[%x], is not HARDWARE_MODULE_TAG[%x] fail!\r\n", hmi->tag, HARDWARE_MODULE_TAG);
			return -1;
		}
		if (hmi->hal_api_version != HARDWARE_HAL_API_VERSION)
		{
			HW_HAL_LOG_UNMASK("check hal_api_version[%x], is not HARDWARE_HAL_API_VERSION[%x] fail!\r\n", hmi->hal_api_version, HARDWARE_HAL_API_VERSION);
			return -1;
		}
		if (!HW_CHECK_MAJ_VERSION(hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION))
		{
			HW_HAL_LOG_UNMASK("check major version is not the same, global_devicetype_version[%x], HW_GLOBAL_DEVTYPE_VERSION[%x]!\r\n", hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION);
			return -1;
		}
		if (hw_global_devtype_magic_get(hmi->devicetype) != hmi->devtype_magic)
		{
			HW_HAL_LOG_UNMASK("check devtype magic fail!\r\n");
			return -1;
		}
		if (!HW_CHECK_MAJ_VERSION(hmi->device_moduleid_version, HW_DEVICE_MODULEID_VERSION))
		{
			HW_HAL_LOG_UNMASK("check major version is not the same, device_moduleid_version[%x], HW_DEVICE_MODULEID_VERSION[%x]!\r\n", hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION);
			return -1;
		}
		HW_HAL_LOG_UNMASK("Check all of the tag and version of the module success!\r\n");
		HW_HAL_LOG_UNMASK("Finish first get the module[%s] success!\r\n", hmi->description);
	}
	*o_ppmodule = hmi;
	return 0;
}

s32 hw_module_put(struct hw_module_t* i_pmodule)
{
	dlclose(i_pmodule->privdata.dso);
	return 0;
}

s32 hw_module_device_get(struct hw_module_t* i_pmodule, void* i_param, struct hw_device_t** o_ppdevice)
{
	s32 ret = i_pmodule->privapi.device_get(i_pmodule, i_param, o_ppdevice);
	/*
	* We check the output device version here. When not match, put the device back and return <0.
	*/
	if (ret >= 0)
	{
		u32 device_api_version;
		switch (i_pmodule->devicetype)
		{
		case HW_DEVICETYPE_VIDEO:
			device_api_version = HW_VIDEO_API_VERSION;
			break;
		case HW_DEVICETYPE_LIDAR:
			device_api_version = HW_LIDAR_API_VERSION;
			break;
		default:
			HW_HAL_LOG_UNMASK("unexpected devicetype[%d]!\r\n", i_pmodule->devicetype);
			return -1;
		}
		if ((*o_ppdevice)->device_api_version != device_api_version)
		{
			HW_HAL_LOG_UNMASK("device get version[%x] check_device_api_version[%x] fail!\r\n", (*o_ppdevice)->device_api_version, device_api_version);
			return -1;
		}
		return 0;
	}
	return -1;
}

s32 hw_module_device_put(struct hw_module_t* i_pmodule, struct hw_device_t* i_pdevice)
{
	return i_pmodule->privapi.device_put(i_pmodule, i_pdevice);
}
