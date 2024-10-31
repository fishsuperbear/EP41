#include "hw_nvmedia_common_impl.h"

hw_ret_s32 HWNvmediaContext::Init()
{
    CHK_LOG_SENTENCE_HW_RET_S32(checkversion());
    CHK_LOG_SENTENCE_HW_RET_S32(queryprepare());
    return 0;
}

hw_ret_s32 HWNvmediaContext::queryprepare()
{
    SIPLStatus status;
    _pquery = INvSIPLQuery::GetInstance();
    CHK_PTR_AND_RET_S32(_pquery, "INvSIPLQuery::GetInstance");
    CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(status = _pquery->ParseDatabase(), "INvSIPLQuery::ParseDatabase");
    return 0;
}

hw_ret_s32 HWNvmediaContext::checkversion()
{
    NvSIPLVersion oVer;
    NvSIPLGetVersion(oVer);

    HW_NVMEDIA_LOG_INFO("NvSIPL library version: %u.%u.%u\n", oVer.uMajor, oVer.uMinor, oVer.uPatch);
    HW_NVMEDIA_LOG_INFO("NVSIPL header version: %u %u %u\n", NVSIPL_MAJOR_VER, NVSIPL_MINOR_VER, NVSIPL_PATCH_VER);
    if (oVer.uMajor != NVSIPL_MAJOR_VER || oVer.uMinor != NVSIPL_MINOR_VER || oVer.uPatch != NVSIPL_PATCH_VER) {
        HW_NVMEDIA_LOG_FATAL("NvSIPL library and header version mismatch!\r\n");
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_CHECK_VERSION_FAIL);
    }
    else
    {
        HW_NVMEDIA_LOG_INFO("NvSIPL library and header version match!\r\n");
    }
    return 0;
}
