#include "hw_nvmedia_common_impl.h"

SIPLStatus HWNvmediaContext::checksku(const std::string& findStr, bool& bFound)
{
    std::string sTargetModelNode = "/proc/device-tree/model";
    std::ifstream fs;
    fs.open(sTargetModelNode, std::ifstream::in);
    if (!fs.is_open()) {
        HW_NVMEDIA_LOG_ERR("%s: cannot open board node %s\r\n", __FUNCTION__, sTargetModelNode.c_str());
        return NVSIPL_STATUS_ERROR;
    }

    // Read the file in to the string.
    std::string nodeString;
    fs >> nodeString;

    if (strstr(nodeString.c_str(), findStr.c_str())) {
        HW_NVMEDIA_LOG_INFO("CheckSKU found findStr!\r\n");
        bFound = true;
    }
    else {
        HW_NVMEDIA_LOG_INFO("CheckSKU not found findStr.\r\n");
    }

    if (fs.is_open()) {
        fs.close();
    }
    return NVSIPL_STATUS_OK;
}

hw_ret_s32 HWNvmediaContext::updateplatformcfg_perboardmodel()
{
    /* GPIO power control is required for Drive Orin (P3663) but not Firespray (P3710)
     * If using another platform (something customer-specific, for example) the GPIO
     * field may need to be modified
     */
    bool isP3663 = false;
    CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(checksku("3663", isP3663), "CheckSKU");
    if (isP3663) {
        std::vector<uint32_t> gpios = { 7 };
        CHK_PTR_AND_RET_S32_BADARG(_platformcfg.deviceBlockList, "deviceBlockList");
        _platformcfg.deviceBlockList[0].gpios = gpios;
    }
    return 0;
}

hw_ret_s32 HWNvmediaContext::deviceopen()
{
#if (HW_NVMEDIA_MAY_CHANGE_ME_LATER == 1)
    /*
    * Currently, we will ensure one process only one thread use the device.
    * Later we may add logic to ensure only one process use the device.
    */
    if (hw_plat_atomic_cas_exchange_u32(&_busingdevice, 0, 1, NULL) < 0)
    {
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_DEVICE_ALREADY_IN_USE);
    }
#endif
    CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(_pquery->GetPlatformCfg(_deviceopenpara.platformname, _platformcfg),
        "INvSIPLQuery::GetPlatformCfg");
    HW_NVMEDIA_LOG_INFO("Setting link masks\r\n");
    CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(_pquery->ApplyMask(_platformcfg, _deviceopenpara.vmasks),
        "INvSIPLQuery::ApplyMask");
    CHK_LOG_SENTENCE_HW_RET_S32(updateplatformcfg_perboardmodel());
    /*
    * Set member variables according to platform cfg.
    */
    _numblocks = _platformcfg.numDeviceBlocks;
    u32 blocki;
    u32 sensori;
    u32 outputi;
    DeviceBlockInfo* pdeviceblockinfo;
    for (blocki = 0; blocki < _numblocks; blocki++)
    {
        pdeviceblockinfo = &_platformcfg.deviceBlockList[blocki];
        _parrayblockinfo[blocki].numsensors_byblock = pdeviceblockinfo->numCameraModules;
        for (sensori = 0; sensori < pdeviceblockinfo->numCameraModules; sensori++)
        {
            _vcameramodules.push_back(pdeviceblockinfo->cameraModuleInfoList[sensori]);
            /*
            * New profiler instances.
            */
            for (outputi = 0; outputi < HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_COUNT; outputi++)
            {
                _ppparraypoutputpipelineprofiler[blocki][sensori][outputi] = new HWNvmediaSensorOutputPipelineProfiler(
                    blocki, sensori, pdeviceblockinfo->cameraModuleInfoList[sensori].sensorInfo.id,
                    (HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE)outputi);
            }
        }
    }
    _numsensors = (u32)_vcameramodules.size();
    HW_NVMEDIA_LOG_UNMASK("_numblocks = %u, _numsensors = %u\r\n", _numblocks, _numsensors);
    _pvideoinfoext = NULL;	// currently always NULL, reserved for future
    CHK_LOG_SENTENCE_HW_RET_S32(ModuleOpen());
    return 0;
}

hw_ret_s32 HWNvmediaContext::videoinfoget(struct hw_video_info_t* o_pinfo)
{
    if (o_pinfo)
    {
        o_pinfo->numblocks = _numblocks;
        o_pinfo->numsensors = _numsensors;
        u32 blocki;
        for (blocki = 0; blocki < _numblocks; blocki++)
        {
            o_pinfo->parrayblockinfo[blocki] = _parrayblockinfo[blocki];
        }
        o_pinfo->pext = _pvideoinfoext;   // currently it is always NULL
    }
    return 0;
}

hw_ret_s32 HWNvmediaContext::deviceclose()
{
    // need to release _pvideoinfoext when _pvideoinfoext is not NULL in the future
    u32 blocki;
    u32 sensori;
    u32 outputi;
    DeviceBlockInfo* pdeviceblockinfo;
    CHK_LOG_SENTENCE_HW_RET_S32(ModuleClose());
    for (blocki = 0; blocki < _numblocks; blocki++)
    {
        pdeviceblockinfo = &_platformcfg.deviceBlockList[blocki];
        _parrayblockinfo[blocki].numsensors_byblock = pdeviceblockinfo->numCameraModules;
        for (sensori = 0; sensori < pdeviceblockinfo->numCameraModules; sensori++)
        {
            /*
            * Delete profiler instances.
            */
            for (outputi = 0; outputi < HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_COUNT; outputi++)
            {
                delete(_ppparraypoutputpipelineprofiler[blocki][sensori][outputi]);
            }
        }
    }
    _vcameramodules.clear();
    _numblocks = 0;
    _numsensors = 0;
    _numoutputpipeline = 0;
    if (hw_plat_atomic_cas_exchange_u32(&_busingdevice, 1, 0, NULL) < 0)
    {
        HW_NVMEDIA_LOG_ERR("Unexpected exchange fail!\r\n");
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_DEVICE_CLOSE_EXCHANGE_FAIL);
    }
    return 0;
}
