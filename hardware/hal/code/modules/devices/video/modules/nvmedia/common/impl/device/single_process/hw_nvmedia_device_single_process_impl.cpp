#include "hw_nvmedia_common_impl.h"

hw_ret_s32 HWNvmediaSingleProcessContext::Device_Open(struct hw_video_info_t* o_pinfo)
{
    HW_NVMEDIA_LOG_INFO("Device Open Enter!\r\n");
	CHK_LOG_SENTENCE_HW_RET_S32(deviceopen());
	CHK_LOG_SENTENCE_HW_RET_S32(videoinfoget(o_pinfo));
	return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::Device_Close()
{
	HW_NVMEDIA_LOG_INFO("Device Close Enter!\r\n");
	CHK_LOG_SENTENCE_HW_RET_S32(deviceclose());
	return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::ModuleOpen()
{
	_pcamera = INvSIPLCamera::GetInstance();
	CHK_PTR_AND_RET_S32(_pcamera, "INvSIPLCamera::GetInstance()");
	CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciBufModuleOpen(&_scibufmodule), "NvSciBufModuleOpen");
	CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciSyncModuleOpen(&_scisyncmodule), "NvSciSyncModuleOpen");
	// single process do not need to call NvSciIpcInit
	return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::ModuleClose()
{
	if (_scibufmodule != nullptr) {
		NvSciBufModuleClose(_scibufmodule);
		_scibufmodule = nullptr;
	}
	if (_scisyncmodule != nullptr) {
		NvSciSyncModuleClose(_scisyncmodule);
		_scisyncmodule = nullptr;
	}
	return 0;
}
