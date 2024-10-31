#include "hw_nvmedia_compile.h"

#if (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_GROUPA)
#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_groupa_impl.h"

extern "C" struct hw_nvmedia_groupa_module_t HAL_MODULE_INFO_SYM;
struct hw_nvmedia_groupa_module_t HAL_MODULE_INFO_SYM =
{
	.common = {
		.tag = HARDWARE_MODULE_TAG,
		.hal_api_version = HARDWARE_HAL_API_VERSION,
		.global_devicetype_version = HW_GLOBAL_DEVTYPE_VERSION,
		.devicetype = HW_DEVICETYPE_VIDEO,
		.module_id = HW_VIDEO_MODULEID_NVMEDIA_GROUPA,
		.devtype_magic = hw_global_devtype_magic_get(HW_DEVICETYPE_VIDEO),
		.device_moduleid_version = HW_DEVICE_MODULEID_VERSION,
		.module_api_version = HW_MODULE_API_VERSION,
		.devmoduleid_magic = hw_video_moduleid_magic_get(HW_VIDEO_MODULEID_NVMEDIA_GROUPA),
		.description = "groupa block",
		.privdata = {
			.dso = NULL,
			.pvoid = NULL,
		},
		.privapi = {
			.init = hw_module_privapi_init,
			.check_device_api_version = hw_module_privapi_check_device_api_version,
			.trigger_check_unmasklogleft = hw_module_privapi_trigger_check_unmasklogleft,
			.device_get = hw_module_privapi_device_get,
			.device_put = hw_module_privapi_device_put,
		},
	},
	.nvmedia = {
		.driveosversiondesc = "6.0.5",
	},
};

static HWNvmediaGroupaEnv		_env;

s32 hw_module_privapi_init(void** io_ppvoid)
{
	if (_env.binit == 0)
	{
		HW_NVMEDIA_LOG_UNMASK("Run init of module[%s]nvmedia-driveosversiondesc[%s]!\r\n", HAL_MODULE_INFO_SYM.common.description, 
			HAL_MODULE_INFO_SYM.nvmedia.driveosversiondesc);
		/*
		* Set all of the global .data content the init value.
		*/

		/*
		* For future reserved.
		*/
		*io_ppvoid = NULL;

		_env.binit = 1;
		return 0;
	}
	return -1;
}

s32 hw_module_privapi_check_device_api_version(u32 i_device_api_version)
{
	_env.device_api_version = i_device_api_version;
	/*
	* The module currently only support HARDWARE_DEVICE_API_VERSION(0, 1).
	*/
	if (i_device_api_version == HW_MAKEV_DEVICE_API_VERSION(0, 1))
	{
		_env.bsupport_device_api_version = 1;
		HW_NVMEDIA_LOG_UNMASK("The current module support the device_api_version[%x]!\r\n", _env.device_api_version);
		return 0;
	}
	else
	{
		_env.bsupport_device_api_version = 0;
		HW_NVMEDIA_LOG_UNMASK("The current module does not support the device_api_version[%x]!\r\n", _env.device_api_version);
		return -1;
	}
}

s32 hw_module_privapi_trigger_check_unmasklogleft()
{
	/*
	* For future use.
	*/
	return 0;
}

s32 hw_module_privapi_device_get(struct hw_module_t* i_pmodule, void* i_param, struct hw_device_t** io_ppdevice)
{
	/*
	* Inner use only. Only for nvmedia internal api currently.
	*/
	hw_ret_s32 ret;
	/*
	* First need to check whether support the current device api version. We already check it 
	* when call hw_module_get in the previous step.
	*/
	if (HW_UNLIKELY(_env.bsupport_device_api_version == 0))
	{
		HW_NVMEDIA_LOG_UNMASK("bsupport_device_api_version is 0, so cannot get device!\r\n");
		return -1;
	}
	struct hw_video_t* pvideo = (struct hw_video_t*)malloc(sizeof(struct hw_video_t));
	pvideo->common.tag = HARDWARE_DEVICE_TAG;
	pvideo->common.device_api_version = _env.device_api_version;
	pvideo->common.pmodule = (hw_module_t*)&HAL_MODULE_INFO_SYM;
	hw_nvmedia_setvideoops(pvideo);
	/*
	* Check get pcontext.
	*/
	hw_plat_mutex_lock(&_env.mutex_context);
	if (_env.pcontext == NULL)
	{
		_env.pcontext = new HWNvmediaGroupaContext(pvideo);
		ret = _env.pcontext->Init();
		if (ret != 0)
		{
			HW_NVMEDIA_LOG_UNMASK("ContextPrepare fail!\r\n");
			delete(_env.pcontext);
			hw_plat_mutex_unlock(&_env.mutex_context);
			return -2;
		}
		_env.refcount_pcontext = 0;
	}
	pvideo->priv = _env.pcontext;
	_env.refcount_pcontext++;
	hw_plat_mutex_unlock(&_env.mutex_context);
	HW_NVMEDIA_LOG_INFO("Has init nvmedia groupa context!\r\n");
	HW_NVMEDIA_LOG_INFO("The current drive os version[%s]\r\n", HAL_MODULE_INFO_SYM.nvmedia.driveosversiondesc);
	*io_ppdevice = (struct hw_device_t*)pvideo;
	return 0;
}

s32 hw_module_privapi_device_put(struct hw_module_t* i_pmodule, struct hw_device_t* i_pdevice)
{
	struct hw_video_t* pvideo = (struct hw_video_t*)i_pdevice;
	if (HW_UNLIKELY(pvideo->common.tag != HARDWARE_DEVICE_TAG))
	{
		HW_NVMEDIA_LOG_UNMASK("unexpected video device tag[%x], should be HARDWARE_DEVICE_TAG[%x]\r\n", pvideo->common.tag, HARDWARE_DEVICE_TAG);
		return -1;
	}
	HWNvmediaGroupaContext* pcontext = (HWNvmediaGroupaContext*)pvideo->priv;
	delete(pcontext);
	delete(pvideo);
	return 0;
}

#endif
