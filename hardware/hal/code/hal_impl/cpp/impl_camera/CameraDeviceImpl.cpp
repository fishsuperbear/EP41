#include <mutex>
#include "hal_camera_impl.h"

static CameraDeviceImplEnv _main_env;

std::unique_ptr<ICameraDevice> ICameraDevice::GetInstance(HAL_CAMERA_VERSION i_version)
{
    hw_plat_mutex_lock(&_main_env.mutex_global);
    if (_main_env.globalenv.bhasinstance != 0) {
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        /*
         * One process can own one instance at one time.
         */
        HAL_CAMERA_LOG_ERR("Failed! One process can own one instance at one time!\r\n");
        return std::unique_ptr<ICameraDevice>(_main_env.globalenv.pcameradevice);
    }else{
        _main_env.globalenv.bhasinstance = 1;
        _main_env.globalenv.pcameradevice = new CameraDeviceImpl();
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        return std::unique_ptr<ICameraDevice>(_main_env.globalenv.pcameradevice);
    }
}

s32 CameraDeviceImpl::CreateCameraSession(ICameraDeviceCallback* i_pcallback, ICameraDeviceSession** o_ppsession)
{
    s32 ret;
    CameraDeviceSessionImpl* pdevicesession;
    const char* modulename;
    hw_module_t* pmodule;
    hw_video_t* pvideo;
    hw_plat_mutex_lock(&_main_env.mutex_global);
    CAMERA_DEVICE_OPENTYPE opentype = i_pcallback->RegisterOpenType();
    switch (opentype)
    {
        case CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY:
            if(i_pcallback->RegisterOpenMode() == CAMERA_DEVICE_OPENMODE_MAIN){
                modulename = "libhw_nvmedia_multiipc_main.so";
            }else{
                modulename = "libhw_nvmedia_multiipc_cuda.so";
            }
            break;
        default:
            hw_plat_mutex_unlock(&_main_env.mutex_global);
            /*
             * Currently do not support the opentype.
             * The following opentype is to be supported:
             */
            HAL_CAMERA_LOG_ERR("Failed! Currently do not support the opentype[%u]!\r\n", opentype);
            *o_ppsession = nullptr;
            return -3;
    }
    /*
     * Get the hw_module_t and hw_video_t first by opentype.
     */
    ret = hw_module_get(modulename, &pmodule);
    if (ret < 0) {
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        HAL_CAMERA_LOG_ERR("hw_module_get fail! ret=%x\r\n", ret);
        *o_ppsession = nullptr;
        return ret;
    }
    ret = hw_module_device_get(pmodule, NULL, (hw_device_t**)&pvideo);
    if (ret < 0) {
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        HAL_CAMERA_LOG_ERR("hw_module_device_get fail! ret=%x\r\n", ret);
        *o_ppsession = nullptr;
        return ret;
    }
    _pvideoprocessors[_sessionnum].pmodule = pmodule;
    _pvideoprocessors[_sessionnum].pvideo = pvideo;
    if (_parray_psession[_sessionnum] == nullptr) {
        pdevicesession = new CameraDeviceSessionImpl(_sessionnum, pvideo);
    }
    else {
        pdevicesession = _parray_psession[_sessionnum];
        pdevicesession->ReInit(_sessionnum, pvideo);
    }
    _psession_toinit = pdevicesession;
    if ((ret = pdevicesession->Internal_Init(i_pcallback)) < 0) {
        _psession_toinit = nullptr;
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        HAL_CAMERA_LOG_ERR("Camera Device Session Internal_Init fail! ret=%x\r\n", ret);
        *o_ppsession = nullptr;
        return ret;
    }
    _psession_toinit = nullptr;
    _parray_psession[_sessionnum++] = pdevicesession;
    *o_ppsession = pdevicesession;
    hw_plat_mutex_unlock(&_main_env.mutex_global);
    return 0;
}

void CameraDeviceImpl::Internal_LowerLevelDataCb(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
    CameraDeviceDataCbEnv* pdatacbenv = (CameraDeviceDataCbEnv*)i_pbufferinfo->pcustom;
    /*
     * Currently the CameraDeviceDataCbInfo gpubufferinfo is set when CameraDeviceImpl::RegisterDataCallback.
     * We only need to change its buffer pointer and size here.
     * We may dynamically set other info according to the i_pcudabufferinfo in the future version.
     */
    CameraDeviceDataCbInfo bufferinfo = pdatacbenv->bufferinfo;
    //switch(i_pbufferinfo->blockindex)
    //{
    //    case 0:
    //        bufferinfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPA;
    //    case 1:
    //        bufferinfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPB;
    //    case 2:
    //        bufferinfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPC;
    //    case 3:
    //    default:
    //        bufferinfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPD;
    //        break;
    //}
    //bufferinfo.sensorindex = i_pbufferinfo->sensorindex;
    bufferinfo.capturewidth = i_pbufferinfo->capturewidth;
    bufferinfo.captureheight = i_pbufferinfo->captureheight;
    bufferinfo.width = i_pbufferinfo->width;
    bufferinfo.height = i_pbufferinfo->height;
    bufferinfo.rotatedegrees = i_pbufferinfo->rotatedegrees;
    bufferinfo.pbuff = i_pbufferinfo->pbuff;
    bufferinfo.size = i_pbufferinfo->size;
    bufferinfo.timeinfo.timestamp = i_pbufferinfo->timeinfo.exposurestarttime;
    /*
     * The data callback may run more than one simultaneously, so use stack memory here.
     */
    pdatacbenv->datacb(&bufferinfo);
}

void CameraDeviceImpl::Internal_LowerLevelGpuDataCb(struct hw_video_cudabufferinfo_t* i_pcudabufferinfo)
{
    CameraDeviceDataCbEnv* pdatacbenv = (CameraDeviceDataCbEnv*)i_pcudabufferinfo->pcustom;
    /*
     * We ensure the gpu data callback is handled one by one, so we use the pointer of the single instance.
     */
    /*
     * Currently the CameraDeviceDataCbInfo gpubufferinfo is set when CameraDeviceImpl::RegisterDataCallback.
     * We only need to change its buffer pointer and size here.
     * We may dynamically set other info according to the i_pcudabufferinfo in the future version.
     */
    CameraDeviceGpuDataCbInfo* pgpubufferinfo = &pdatacbenv->gpubufferinfo;
    //switch (i_pcudabufferinfo->blockindex) {
    //    case 0:
    //        pgpubufferinfo->blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPA;
    //    case 1:
    //        pgpubufferinfo->blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPB;
    //    case 2:
    //        pgpubufferinfo->blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPC;
    //    case 3:
    //    default:
    //        pgpubufferinfo->blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPD;
    //        break;
    //}
    //pgpubufferinfo->sensorindex = i_pcudabufferinfo->sensorindex;
    pgpubufferinfo->capturewidth = i_pcudabufferinfo->capturewidth;
    pgpubufferinfo->captureheight = i_pcudabufferinfo->captureheight;
    pgpubufferinfo->width = i_pcudabufferinfo->width;
    pgpubufferinfo->height = i_pcudabufferinfo->height;
    pgpubufferinfo->stride = i_pcudabufferinfo->stride;
    pgpubufferinfo->rotatedegrees = i_pcudabufferinfo->rotatedegrees;
    if (CameraDeviceImpl::IsRGBType_ByGPUDataCbImgType(pgpubufferinfo->pgpuinfo->rgbtype))
    {
        pgpubufferinfo->pgpuinfo->prgbinfo->pbuff = i_pcudabufferinfo->gpuinfo.rgbinfo.pbuff;
        pgpubufferinfo->pgpuinfo->prgbinfo->buffsize = i_pcudabufferinfo->gpuinfo.rgbinfo.buffsize;
    }
    pgpubufferinfo->timeinfo.timestamp = i_pcudabufferinfo->timeinfo.exposurestarttime;
    /*
     * We ensure the gpu data callback is handled one by one, so we use the pointer of the single instance.
     */
    pdatacbenv->gpudatacb(pgpubufferinfo);
}

s32 CameraDeviceImpl::RegisterEventCallback(CameraDeviceEventCbRegInfo* i_peventreginfo, camera_device_eventcb i_eventcb)
{
    /*
     * Within mutex_global lock. We get the psession pointer to init.
     */
    /*
     * Check opentype here.
     */
    /* CameraDeviceSessionImpl* o_session = nullptr; */
    /* for(int i =0;i<CAMERADEVICE_SESSION_MAXNUM;i++){ */
    /*     if(_parray_psession[i]!=nullptr){ */
    /*         if(_parray_psession[i]->_openmode == CAMERA_DEVICE_OPENMODE_MAIN */
    /*                 && i_peventreginfo->opentype == _parray_psession[i]->_opentype){ */
    /*             o_session = _parray_psession[i]; */
    /*             break; */
    /*         } */
    /*     } */
    /* } */
    /* if (o_session==nullptr) { */
    /*     return -1; */
    /* } */
    u32 ret;
    u32 blockindex;
    u32 sensorindex;
    if ((ret = GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex(_psession_toinit->_opentype,
                    i_peventreginfo->blocktype, i_peventreginfo->sensorindex, &blockindex, &sensorindex)) < 0) {
        HAL_CAMERA_LOG_ERR("GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex fail in RegisterEventCallback!\r\n");
        return -1;
    }
    _psession_toinit->_parray_eventcb[blockindex][sensorindex] = i_eventcb;
    /* o_session = nullptr; */
    return 0;
}

s32 CameraDeviceImpl::RegisterCpuDataCallback(const CameraDeviceCpuDataCbRegInfo* i_pdatareginfo, camera_device_datacb i_datacb)
{
    /*
     * Within mutex_global lock. We get the psession pointer to init.
     */
    /*
     * Check opentype here.
     */
    // CameraDeviceSessionImpl* o_session = nullptr;
    /* for(int i =0;i<CAMERADEVICE_SESSION_MAXNUM;i++){ */
    /*     if(_parray_psession[i]!=nullptr){ */
    /*         if(_parray_psession[i]->_openmode == CAMERA_DEVICE_OPENMODE_SUB){ */
    /*                 /1* && i_pdatareginfo->opentype == _parray_psession[i]->_opentype){ *1/ */
    /*             o_session = _parray_psession[i]; */
    /*             break; */
    /*         } */
    /*     } */
    /* } */
    /* if (o_session==nullptr) { */
    /*     printf("register RegisterCpuDataCallback failed\n"); */
    /*     return -1; */
    /* } */
    u32 ret;
    u32 blockindex;
    u32 sensorindex;
    if ((ret = GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex(_psession_toinit->_opentype,
                    i_pdatareginfo->blocktype, i_pdatareginfo->sensorindex, &blockindex, &sensorindex)) < 0) {
        HAL_CAMERA_LOG_ERR("GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex fail in RegisterDataCallback!\r\n");
        return -1;
    }
    u32 datacbindex = _psession_toinit->_parray_datacbnum[blockindex][sensorindex];
    CameraDeviceDataCbEnv* pdatacbenv = &_psession_toinit->_parray_datacbenv[blockindex][sensorindex][datacbindex];
    pdatacbenv->bgpucb = 0;
    pdatacbenv->datacbreginfo = *i_pdatareginfo;
    pdatacbenv->datacb = i_datacb;
    /*
     * Currently set buffer info according to the register info.
     * We may dynamically set it when pipeline run in the future version.
     */
    pdatacbenv->bufferinfo.opentype = _psession_toinit->_opentype;
    pdatacbenv->bufferinfo.blocktype = i_pdatareginfo->blocktype;
    pdatacbenv->bufferinfo.sensorindex = i_pdatareginfo->sensorindex;
    pdatacbenv->bufferinfo.datacbtype = i_pdatareginfo->datacbtype;
    pdatacbenv->bufferinfo.pcontext = i_pdatareginfo->pcontext;
    pdatacbenv->lowerlevelblockindex = blockindex;
    pdatacbenv->lowerlevelsensorindex = sensorindex;
    if ((ret = GetLowerLevelDataCbType_ByDataCbType(i_pdatareginfo->datacbtype, &pdatacbenv->lowerlevelregdatacbtype)) < 0) {
        HAL_CAMERA_LOG_ERR("GetLowerLevelDataCbType_ByDataCbType fail in RegisterDataCallback!\r\n");
        return -1;
    }
    pdatacbenv->lowerleveldatacb = Internal_LowerLevelDataCb;
    _psession_toinit->_parray_datacbnum[blockindex][sensorindex] = datacbindex + 1;
    HAL_CAMERA_LOG_INFO("CameraDeviceImpl::RegisterDataCallback end;blockidx=%d,sensoridx=%d ,cb num=%d\n",blockindex,sensorindex,_psession_toinit->_parray_datacbnum[blockindex][sensorindex]);
    /* o_session = nullptr; */
    return 0;
}

s32 CameraDeviceImpl::RegisterGpuDataCallback(const CameraDeviceGpuDataCbRegInfo* i_pgpureginfo, camera_device_gpudatacb i_gpudatacb)
{
    /*
     * Within mutex_global lock. We get the psession pointer to init.
     */
    /*
     * Check opentype here.
     */
    // CameraDeviceSessionImpl* o_session = nullptr;
    /* for(int i =0;i<CAMERADEVICE_SESSION_MAXNUM;i++){ */
    /*     if(_parray_psession[i]!=nullptr){ */
    /*         if(_parray_psession[i]->_openmode == CAMERA_DEVICE_OPENMODE_SUB){ */
    /*                 /1* && i_pgpureginfo->opentype == _parray_psession[i]->_opentype){ *1/ */
    /*             o_session = _parray_psession[i]; */
    /*             break; */
    /*         } */
    /*     } */
    /* } */
    /* if (o_session==nullptr) { */
    /*     printf("register RegisterGpuDataCallback failed\n"); */
    /*     return -1; */
    /* } */
    u32 ret;
    u32 blockindex;
    u32 sensorindex;
    if ((ret = GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex(_psession_toinit->_opentype,
                    i_pgpureginfo->blocktype,  i_pgpureginfo->sensorindex, &blockindex, &sensorindex)) < 0) {
        HAL_CAMERA_LOG_ERR("GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex fail in RegisterGpuDataCallback!\r\n");
        return -1;
    }
    u32 datacbindex = _psession_toinit->_parray_datacbnum[blockindex][sensorindex];
    CameraDeviceDataCbEnv* pdatacbenv = &_psession_toinit->_parray_datacbenv[blockindex][sensorindex][datacbindex];
    pdatacbenv->bgpucb = 1;
    pdatacbenv->gpudatacbreginfo = *i_pgpureginfo;
    pdatacbenv->gpudatacb = i_gpudatacb;
    /*
     * Currently set buffer info according to the register info.
     * We may dynamically set it when pipeline run in the future version.
     */
    pdatacbenv->gpubufferinfo.opentype = _psession_toinit->_opentype;
    pdatacbenv->gpubufferinfo.blocktype = i_pgpureginfo->blocktype;
    pdatacbenv->gpubufferinfo.sensorindex = i_pgpureginfo->sensorindex;
    pdatacbenv->gpubufferinfo.gpuimgtype = i_pgpureginfo->gpuimgtype;
    pdatacbenv->gpubufferinfo.interpolation = i_pgpureginfo->interpolation;
    pdatacbenv->gpubufferinfo.pcontext = i_pgpureginfo->pcontext;
    pdatacbenv->gpubufferinfo.pgpuinfo = &pdatacbenv->gpubufferinfo_buffinfo;
    pdatacbenv->gpubufferinfo.pgpuinfo->rgbtype = i_pgpureginfo->gpuimgtype;
    pdatacbenv->gpubufferinfo.pgpuinfo->prgbinfo = nullptr;
    pdatacbenv->gpubufferinfo.pgpuinfo->pyuvinfo = nullptr;
    if (IsRGBType_ByGPUDataCbImgType(i_pgpureginfo->gpuimgtype))
    {
        pdatacbenv->gpubufferinfo.pgpuinfo->prgbinfo = &pdatacbenv->gpubufferinfo_rgbinfo;
        pdatacbenv->gpubufferinfo.pgpuinfo->prgbinfo->imgtype = i_pgpureginfo->gpuimgtype;
    }
    if (IsYUVType_ByGPUDataCbImgType(i_pgpureginfo->gpuimgtype))
    {
        pdatacbenv->gpubufferinfo.pgpuinfo->pyuvinfo = &pdatacbenv->gpubufferinfo_yuvinfo;
    }
    pdatacbenv->lowerlevelblockindex = blockindex;
    pdatacbenv->lowerlevelsensorindex = sensorindex;
    pdatacbenv->lowerlevelregdatacbtype = HW_VIDEO_REGDATACB_TYPE_CUDA;
    pdatacbenv->lowerlevelcudadatacb = Internal_LowerLevelGpuDataCb;
    _psession_toinit->_parray_datacbnum[blockindex][sensorindex] = datacbindex + 1;
    return 0;
}

CameraDeviceImpl::~CameraDeviceImpl()
{
    /*
    * Delete the session pointers.
    */
    u32 sessioni;
    for (sessioni = 0; sessioni < CAMERADEVICE_SESSION_MAXNUM; sessioni++)
    {
        if (_parray_psession[sessioni] != nullptr) {
            delete _parray_psession[sessioni];
        }
    }
}

s32 CameraDeviceImpl::Internal_CloseSession(u32 i_sessionindex)
{
    s32 ret;
    hw_plat_mutex_lock(&_main_env.mutex_global);
    /*
    * Currently only support one session.
    */
    if (i_sessionindex != 0) {
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        HAL_CAMERA_LOG_ERR("Internal_CloseSession fail! Unexpected i_sessionindex=%u\r\n", i_sessionindex);
        return -1;
    }
    if ((ret = hw_module_device_put(_pvideoprocessors[i_sessionindex].pmodule, (hw_device_t*)_pvideoprocessors[i_sessionindex].pvideo)) < 0) {
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        HAL_CAMERA_LOG_ERR("hw_module_device_put fail! ret=%x\r\n", ret);
        return -1;
    }
    if ((ret = hw_module_put(_pvideoprocessors[i_sessionindex].pmodule)) < 0) {
        hw_plat_mutex_unlock(&_main_env.mutex_global);
        HAL_CAMERA_LOG_ERR("hw_module_put fail! ret=%x\r\n", ret);
        return -1;
    }
    _sessionnum--;
    hw_plat_mutex_unlock(&_main_env.mutex_global);
    /*
     * Close sub session
     */
    return 0;
}

bool CameraDeviceImpl::IsRGBType_ByGPUDataCbImgType(CAMERA_DEVICE_GPUDATACB_IMGTYPE i_imgtype)
{
    switch (i_imgtype)
    {
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_RGB:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_BGR:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_RGB:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_BGR:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_RGB:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_BGR:
        return true;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV12:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV21:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_YUYV:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_YVYU:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_VYUY:
        return false;
    default:
        HAL_CAMERA_LOG_ERR("IsRGBType_ByGPUDataCbImgType fail! Unexpected i_imgtype[%u]\r\n", i_imgtype);
        return false;
    }
}

bool CameraDeviceImpl::IsYUVType_ByGPUDataCbImgType(CAMERA_DEVICE_GPUDATACB_IMGTYPE i_imgtype)
{
    switch (i_imgtype)
    {
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_RGB:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_BGR:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_RGB:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_BGR:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_RGB:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_BGR:
        return false;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV12:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV21:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_YUYV:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_YVYU:
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_VYUY:
        return true;
    default:
        HAL_CAMERA_LOG_ERR("IsRGBType_ByGPUDataCbImgType fail! Unexpected i_imgtype[%u]\r\n", i_imgtype);
        return false;
    }
}

s32 CameraDeviceImpl::GetLowerLevelBlockIndex_ByOpenTypeBlockType(CAMERA_DEVICE_OPENTYPE i_opentype,
    CAMERA_DEVICE_BLOCK_TYPE i_blocktype, u32* o_pblockindex)
{
    switch (i_opentype)
    {
    default:
        HAL_CAMERA_LOG_ERR("GetLowerLevelBlockIndex_ByOpenTypeBlockType fail! Unexpected i_opentype[%u]\r\n", i_opentype);
        return -1;
    }
    return 0;
}

s32 CameraDeviceImpl::GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex(CAMERA_DEVICE_OPENTYPE i_opentype,
    CAMERA_DEVICE_BLOCK_TYPE i_blocktype, u32 i_sensorindex, u32* o_pblockindex, u32* o_psensorindex)
{
    switch (i_opentype)
    {
    case CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY:
        *o_psensorindex = i_sensorindex;
        switch(i_blocktype)
        {
            case CAMERA_DEVICE_BLOCK_TYPE_GROUPA:
                *o_pblockindex = 0;
                break;
            case CAMERA_DEVICE_BLOCK_TYPE_GROUPB:
                *o_pblockindex = 1;
                break;
            case CAMERA_DEVICE_BLOCK_TYPE_GROUPC:
                *o_pblockindex = 2;
                break;
            case CAMERA_DEVICE_BLOCK_TYPE_GROUPD:
                *o_pblockindex = 3;
                break;
            default:
                *o_pblockindex = 0;
                break;
        }
        break;
    default:
        HAL_CAMERA_LOG_ERR("GetLowerLevelBlockSensorIndex_ByOpenTypeBlockSensorIndex fail! Unexpected i_opentype[%u]\r\n", i_opentype);
        return -1;
    }
    return 0;
}

s32 CameraDeviceImpl::GetLowerLevelDataCbType_ByDataCbType(CAMERA_DEVICE_DATACB_TYPE i_datacbtype, HW_VIDEO_REGDATACB_TYPE* o_plowerleveldatacbtype)
{
    switch (i_datacbtype)
    {
    case CAMERA_DEVICE_DATACB_TYPE_RAW12:
        *o_plowerleveldatacbtype = HW_VIDEO_REGDATACB_TYPE_RAW12;
        return 0;
    case CAMERA_DEVICE_DATACB_TYPE_YUV422:
        *o_plowerleveldatacbtype = HW_VIDEO_REGDATACB_TYPE_YUV422;
        return 0;
    case CAMERA_DEVICE_DATACB_TYPE_YUV420:
        *o_plowerleveldatacbtype = HW_VIDEO_REGDATACB_TYPE_YUV420;
        return 0;
    case CAMERA_DEVICE_DATACB_TYPE_YUV420_PRIV:
        *o_plowerleveldatacbtype = HW_VIDEO_REGDATACB_TYPE_YUV420_PRIV;
        return 0;
    case CAMERA_DEVICE_DATACB_TYPE_RGBA:
        *o_plowerleveldatacbtype = HW_VIDEO_REGDATACB_TYPE_RGBA;
        return 0;
    case CAMERA_DEVICE_DATACB_TYPE_AVC:
        *o_plowerleveldatacbtype = HW_VIDEO_REGDATACB_TYPE_AVC;
        return 0;
    case CAMERA_DEVICE_DATACB_TYPE_HEVC:
        *o_plowerleveldatacbtype = HW_VIDEO_REGDATACB_TYPE_HEVC;
        return 0;
    default:
        HAL_CAMERA_LOG_ERR("GetLowerLevelDataCbType_ByDataCbType fail! Unexpected i_datacbtype[%u]\r\n", i_datacbtype);
        return -1;
    }
}

s32 CameraDeviceImpl::GetLowerLevelCudaImgType_ByGpuImageType(CAMERA_DEVICE_GPUDATACB_IMGTYPE i_gpuimgtype, HW_VIDEO_REGCUDADATACB_IMGTYPE* o_plowerlevelcudaimgtype)
{
    switch (i_gpuimgtype)
    {
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_RGB:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NHWC_RGB;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_BGR:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NHWC_BGR;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_RGB:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NCHW_RGB;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW_BGR:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NCHW_BGR;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_RGB:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NCHW16_RGB;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NCHW16_BGR:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NCHW16_BGR;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV12:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_NV12;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_NV21:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_NV21;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_YUYV:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_YUYV;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_YVYU:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_YVYU;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_IMGTYPE_VYUY:
        *o_plowerlevelcudaimgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_VYUY;
        return 0;
    default:
        HAL_CAMERA_LOG_ERR("GetLowerLevelCudaImgType_ByGpuImageType fail! Unexpected i_gpuimgtype[%u]\r\n", i_gpuimgtype);
        return -1;
    }
}

s32 CameraDeviceImpl::GetLowerLevelInterpolation_ByInterpolation(CAMERA_DEVICE_GPUDATACB_INTERPOLATION i_interpolation, HW_VIDEO_REGCUDADATACB_INTERPOLATION* o_plowerlevelinterpolation)
{
    switch (i_interpolation)
    {
    case CAMERA_DEVICE_GPUDATACB_INTERPOLATION_NEAREST:
        *o_plowerlevelinterpolation = HW_VIDEO_REGCUDADATACB_INTERPOLATION_NEAREST;
        return 0;
    case CAMERA_DEVICE_GPUDATACB_INTERPOLATION_BILINEAR:
        *o_plowerlevelinterpolation = HW_VIDEO_REGCUDADATACB_INTERPOLATION_BILINEAR;
        return 0;
    default:
        HAL_CAMERA_LOG_ERR("GetLowerLevelInterpolation_ByInterpolation fail! Unexpected i_interpolation[%u]\r\n", i_interpolation);
        return -1;
    }
}

CameraDeviceImplEnv::CameraDeviceImplEnv()
{
    /*
    * Register default sig handler.
    */
    hw_plat_regsighandler_default();
    hw_plat_mutex_init(&this->mutex_global, HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE);
    this->binit = 1;
    /* hw_plat_mutex_init(&_sub_env.mutex_global, HW_MUTEX_TYPE_PROCESS_PRIVATE_RECURSIVE); */
    /* _sub_env.binit = 1; */
}

CameraDeviceImplEnv::~CameraDeviceImplEnv()
{
    hw_plat_mutex_deinit(&this->mutex_global);
    /* hw_plat_mutex_deinit(&_sub_env.mutex_global); */
    /*
    * Unregister default sig handler.
    */
    hw_plat_unregsighandler_default();
}
