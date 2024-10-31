#ifndef HW_NVMEDIA_GROUPC_IMPL_H
#define HW_NVMEDIA_GROUPC_IMPL_H

#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_groupc.h"

/*
* Groupc context.
* The instance of the class is stored in priv of hw_video_t.
*/
class HWNvmediaGroupcContext : public HWNvmediaSingleProcessContext
{
private:
	HWNvmediaGroupcContext();
public:
	HWNvmediaGroupcContext(struct hw_video_t* i_pvideo);
	~HWNvmediaGroupcContext()
	{

	}
public:
	virtual hw_ret_s32 GetRegDataCbType_OutputType(HW_VIDEO_REGDATACB_TYPE i_regdatacbtype,
		HW_VIDEO_INPUT_TYPE inputType, HWNvmediaEventHandlerRegDataCbConfig* o_pcbconfig);
};

class HWNvmediaGroupcEnv
{
public:
	HWNvmediaGroupcEnv()
	{
		/*
		* Init mutex only once.
		*/
		hw_plat_mutex_init(&mutex_context, HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE);
	}
	~HWNvmediaGroupcEnv()
	{
		hw_plat_mutex_deinit(&mutex_context);
	}
public:
	/*
	* 0 means has not init
	* 1 means has init, set to 1 only by hw_module_privapi_init.
	*/
	u32								binit = 0;
	/*
	* 0 is the init value.
	* Set when check_device_api_version.
	*/
	u32								device_api_version = 0;
	/*
	* 1 means support the current device_api_version
	* 0 means not support the current device_api_version
	*/
	u32								bsupport_device_api_version = 0;
	/*
	* Mutex for singleton context instance.
	*/
	struct hw_mutex_t				mutex_context;
	/*
	* Single instance of one process.
	*/
	HWNvmediaGroupcContext*			pcontext = NULL;
	/*
	* The referene count of pcontext.
	*/
	u32								refcount_pcontext;
};

s32 hw_module_privapi_init(void** io_ppvoid);
s32 hw_module_privapi_check_device_api_version(u32 i_device_api_version);
s32 hw_module_privapi_trigger_check_unmasklogleft();
s32 hw_module_privapi_device_get(struct hw_module_t* i_pmodule, void* i_param, struct hw_device_t** io_ppdevice);
s32 hw_module_privapi_device_put(struct hw_module_t* i_pmodule, struct hw_device_t* i_pdevice);

#endif
