#include "hw_nvmedia_common_impl.h"

hw_ret_s32 HWNvmediaIpcConsumerContext::Device_Open(struct hw_video_info_t* o_pinfo) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::Device_Open Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(deviceopen());
    CHK_LOG_SENTENCE_HW_RET_S32(videoinfoget(o_pinfo));
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::Device_Close() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::Device_Close Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(deviceclose());
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::ModuleOpen() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::ModuleOpen Enter!\r\n");

    _pcamera = INvSIPLCamera::GetInstance();
    CHK_PTR_AND_RET_S32(_pcamera, "INvSIPLCamera::GetInstance()");
    CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciBufModuleOpen(&_scibufmodule), "NvSciBufModuleOpen");
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::ModuleOpen NvSciBufModuleOpen\n");
    CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciSyncModuleOpen(&_scisyncmodule), "NvSciSyncModuleOpen");
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::ModuleOpen NvSciSyncModuleOpen\n");
    CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciIpcInit(), "NvSciIpcInit");
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::ModuleOpen NvSciIpcInit\n");
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::ModuleClose() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::ModuleClose Enter!\r\n");

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