// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <thread>
#include "CEncConsumer.hpp"
#include "nvscibuf.h"

#define MAX_CLIENTS 10

CEncConsumer::CEncConsumer(NvSciStreamBlock handle, u32 uSensor,
                               u32 i_blockindex, u32 i_sensorindex,
                               NvSciStreamBlock queueHandle,
                               uint16_t encodeWidth,
                               uint16_t encodeHeight,
                              int encodeType, void *i_pvicconsumer)
    : CConsumer("EncConsumer", handle, uSensor, queueHandle, ENC_CONSUMER),
  m_encodeWidth(encodeWidth),
  m_encodeHeight(encodeHeight),
  m_encodeType(encodeType),
  _blockindex(i_blockindex),
  _sensorindex(i_sensorindex)
{
        _pvicconsumer = i_pvicconsumer;
        _enc_server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (_enc_server_fd == -1) {
            cerr << "Failed to create socket" << endl;
            return;
        }

        // bind socket
        sprintf(_socket_path, "/tmp/.cam_hal_enc_%d_%d",i_blockindex,i_sensorindex);
        //char cmd[200];
        //sprintf(cmd,"rm -rf %s",_socket_path);

        unlink(_socket_path);
        //int cmdret = system(cmd);
        //HW_NVMEDIA_LOG_UNMASK("system cmd=%d\r\n", cmdret);

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, _socket_path, sizeof(server_addr.sun_path) - 1);
        if (bind(_enc_server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            cerr << "Failed to bind socket" << endl;
            close(_enc_server_fd);
            unlink(_socket_path);
            return;
        }
        std::thread clientEventThread(ProcessEventHandle,this);
        clientEventThread.detach();

}

hw_ret_s32 CEncConsumer::RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
    HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig)
{
    _datacb = i_pcbconfig->cb;
    /*
    * Currently only support sync mode.
    */
    if (!i_pcbconfig->bsynccb)
    {
        HW_NVMEDIA_LOG_ERR("Only support data cb SYNC mode!\r\n");
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_SYNCCB_MODE);
    }
    _bsynccb = 1;
    _pcontext = i_pcbconfig->pcustom;
    _regdatacbtype = (HW_VIDEO_REGDATACB_TYPE)i_pcbconfig->type;
    _expectedoriginsubtype = i_peventhandlercbconfig->commonconsumerconfig.expectedoriginsubtype;
    _bneedgetpixel = i_peventhandlercbconfig->commonconsumerconfig.bneedgetpixel;
    switch (_regdatacbtype)
    {
    case HW_VIDEO_REGDATACB_TYPE_RAW12:
        _usermaintype = HW_VIDEO_BUFFERFORMAT_MAINTYPE_RAW12;
        _usersubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_RAW12;
        break;
    case HW_VIDEO_REGDATACB_TYPE_YUV422:
        _usermaintype = HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV422;
        _usersubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV422;
        break;
    case HW_VIDEO_REGDATACB_TYPE_YUV420:
        _usermaintype = HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV420;
        _usersubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420;
        break;
    case HW_VIDEO_REGDATACB_TYPE_YUV420_PRIV:
        _usermaintype = HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV420;
        _usersubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PRIV;
        break;
    case HW_VIDEO_REGDATACB_TYPE_RGBA:
        _usermaintype = HW_VIDEO_BUFFERFORMAT_MAINTYPE_RGBA;
        _usersubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_RGBA;
        break;
    case HW_VIDEO_REGDATACB_TYPE_AVC:
        _usermaintype = HW_VIDEO_BUFFERFORMAT_MAINTYPE_AVC;
        _usersubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_AVC;
        break;
    case HW_VIDEO_REGDATACB_TYPE_HEVC:
        _usermaintype = HW_VIDEO_BUFFERFORMAT_MAINTYPE_HEVC;
        _usersubtype = HW_VIDEO_BUFFERFORMAT_SUBTYPE_HEVC;
        break;
    default:
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_REGDATACBTYPE_UNEXPECTED);
    }
    return 0;
}

SIPLStatus CEncConsumer::HandleClientInit(void)
{
    /* if (m_consConfig.bFileDump) { */
    if(m_encodeType==HW_VIDEO_REGDATACB_TYPE_AVC && m_consConfig.bFileDump){
        string fileName = "multicast_enc" + to_string(m_uSensorId) + ".h264";
        m_pOutputFile = fopen(fileName.c_str(), "wb");
        PCHK_PTR_AND_RETURN(m_pOutputFile, "Open encoder output file");
    }else if(m_encodeType==HW_VIDEO_REGDATACB_TYPE_HEVC && m_consConfig.bFileDump){
        string fileName = "multicast_enc" + to_string(m_uSensorId) + ".h265";
        m_pOutputFile = fopen(fileName.c_str(), "wb");
        PCHK_PTR_AND_RETURN(m_pOutputFile, "Open encoder output file");
    }
    /* _shmaddr = i_pcbconfig->shmaddr; */
    _shmid = shmget(getShmKeyBySensorID(_blockindex,_sensorindex), SHM_SIZE, 0666|IPC_CREAT);
    if (_shmid == -1) {
        perror("shmget failed");
        _shmaddr = nullptr;
    }else{
        _shmaddr = (uint8_t*)shmat(_shmid, NULL, 0);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::SetEncodeConfig(void)
{
if(m_encodeType==HW_VIDEO_REGDATACB_TYPE_AVC){
    memset(&m_stEncodeConfigH264Params, 0, sizeof(NvMediaEncodeConfigH264));
    m_stEncodeConfigH264Params.h264VUIParameters =
        (NvMediaEncodeConfigH264VUIParams*) calloc(1, sizeof(NvMediaEncodeConfigH264VUIParams));
    CHK_PTR_AND_RETURN(m_stEncodeConfigH264Params.h264VUIParameters, "Alloc h264VUIParameters failed");
    m_stEncodeConfigH264Params.h264VUIParameters->timingInfoPresentFlag = 1;

    // Setting Up Config Params
    m_stEncodeConfigH264Params.features = NVMEDIA_ENCODE_CONFIG_H264_ENABLE_OUTPUT_AUD;
    m_stEncodeConfigH264Params.gopLength = 16;
    m_stEncodeConfigH264Params.idrPeriod = 16;
    m_stEncodeConfigH264Params.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_INTRA_FRAMES;
    m_stEncodeConfigH264Params.adaptiveTransformMode = NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_AUTOSELECT;
    m_stEncodeConfigH264Params.bdirectMode = NVMEDIA_ENCODE_H264_BDIRECT_MODE_DISABLE;
    m_stEncodeConfigH264Params.entropyCodingMode = NVMEDIA_ENCODE_H264_ENTROPY_CODING_MODE_CAVLC;
    m_stEncodeConfigH264Params.quality = NVMEDIA_ENCODE_QUALITY_L0;

    m_stEncodeConfigH264Params.rcParams.rateControlMode = NVMEDIA_ENCODE_PARAMS_RC_CONSTQP;
    m_stEncodeConfigH264Params.rcParams.params.const_qp.constQP.qpIntra = 32;
    m_stEncodeConfigH264Params.rcParams.params.const_qp.constQP.qpInterP = 35;
    m_stEncodeConfigH264Params.rcParams.params.const_qp.constQP.qpInterB = 25;
    m_stEncodeConfigH264Params.rcParams.numBFrames = 0;
    auto nvmediaStatus = NvMediaIEPSetConfiguration(m_pNvMIEP.get(), &m_stEncodeConfigH264Params);
    PCHK_NVMSTATUS_AND_RETURN(nvmediaStatus, "NvMediaIEPSetConfiguration failed");
}else if (m_encodeType==HW_VIDEO_REGDATACB_TYPE_HEVC){
    memset(&m_stEncodeConfigH265Params, 0, sizeof(NvMediaEncodeConfigH265));
    m_stEncodeConfigH265Params.h265VUIParameters =
        (NvMediaEncodeConfigH265VUIParams*) calloc(1, sizeof(NvMediaEncodeConfigH265VUIParams));
    CHK_PTR_AND_RETURN(m_stEncodeConfigH265Params.h265VUIParameters, "Alloc h265VUIParameters failed");
    /* m_stEncodeConfigH265Params.h265VUIParameters->timingInfoPresentFlag = 1; */

    // Setting Up Config Params
    /* we need to set a feature to use UHP in H265, otherwise the performance is only 2/3.
    *   @see https://partners.nvidia.com/bug/viewbug/4122377?siteID=308303
    * */
    m_stEncodeConfigH265Params.encPreset = NVMEDIA_ENC_PRESET_UHP;
    m_stEncodeConfigH265Params.features = NVMEDIA_ENCODE_CONFIG_H265_ENABLE_ULTRA_FAST_ENCODE;
    m_stEncodeConfigH265Params.gopLength = 16;
    m_stEncodeConfigH265Params.idrPeriod = 16;
    m_stEncodeConfigH265Params.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_INTRA_FRAMES;
    m_stEncodeConfigH265Params.quality = NVMEDIA_ENCODE_QUALITY_L0;

    m_stEncodeConfigH265Params.rcParams.rateControlMode = NVMEDIA_ENCODE_PARAMS_RC_CONSTQP;
    m_stEncodeConfigH265Params.rcParams.params.const_qp.constQP.qpIntra = 32;
    m_stEncodeConfigH265Params.rcParams.params.const_qp.constQP.qpInterP = 35;
    m_stEncodeConfigH265Params.rcParams.params.const_qp.constQP.qpInterB = 25;
    m_stEncodeConfigH265Params.rcParams.numBFrames = 0;
    auto nvmediaStatus = NvMediaIEPSetConfiguration(m_pNvMIEP.get(), &m_stEncodeConfigH265Params);
    PCHK_NVMSTATUS_AND_RETURN(nvmediaStatus, "NvMediaIEPSetConfiguration failed");
}
    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::InitEncoder(NvSciBufAttrList bufAttrList)
{
    SIPLStatus status = NVSIPL_STATUS_ERROR;
    if(m_encodeType==HW_VIDEO_REGDATACB_TYPE_AVC){
        NvMediaEncodeInitializeParamsH264 encoderInitParams;
        memset(&encoderInitParams, 0, sizeof(encoderInitParams));
        encoderInitParams.profile = NVMEDIA_ENCODE_PROFILE_AUTOSELECT;
        encoderInitParams.level = NVMEDIA_ENCODE_LEVEL_AUTOSELECT;
        encoderInitParams.encodeHeight = m_encodeHeight;
        encoderInitParams.encodeWidth = m_encodeWidth;
        encoderInitParams.useBFramesAsRef = 0;
        encoderInitParams.frameRateDen = 1;
        encoderInitParams.frameRateNum = 30;
        encoderInitParams.maxNumRefFrames = 1;
        encoderInitParams.enableExternalMEHints = NVMEDIA_FALSE;
        encoderInitParams.enableAllIFrames = NVMEDIA_FALSE;

        m_pNvMIEP.reset(NvMediaIEPCreate(NVMEDIA_IMAGE_ENCODE_H264,    // codec
                    &encoderInitParams,	       // init params
                    bufAttrList,                  // reconciled attr list
                    0,                            // maxOutputBuffering
                    NVMEDIA_ENCODER_INSTANCE_0)); // encoder instance
    }else if(m_encodeType==HW_VIDEO_REGDATACB_TYPE_HEVC){
        NvMediaEncodeInitializeParamsH265 encoderInitParams;
        memset(&encoderInitParams, 0, sizeof(encoderInitParams));
        encoderInitParams.profile = NVMEDIA_ENCODE_PROFILE_AUTOSELECT;
        encoderInitParams.level = NVMEDIA_ENCODE_LEVEL_AUTOSELECT;
        encoderInitParams.encodeHeight = m_encodeHeight;
        encoderInitParams.encodeWidth = m_encodeWidth;
        encoderInitParams.useBFramesAsRef = 0;
        encoderInitParams.frameRateDen = 1;
        encoderInitParams.frameRateNum = 30;
        encoderInitParams.maxNumRefFrames = 1;
        encoderInitParams.enableExternalMEHints = NVMEDIA_FALSE;
        encoderInitParams.enableAllIFrames = NVMEDIA_FALSE;

        m_pNvMIEP.reset(NvMediaIEPCreate(NVMEDIA_IMAGE_ENCODE_HEVC,    // codec
                    &encoderInitParams,	       // init params
                    bufAttrList,                  // reconciled attr list
                    0,                            // maxOutputBuffering
                    NVMEDIA_ENCODER_INSTANCE_0)); // encoder instance

    }
    PCHK_PTR_AND_RETURN(m_pNvMIEP, "NvMediaIEPCreate");

    status = SetEncodeConfig();
    return status;
}

CEncConsumer::~CEncConsumer(void)
{
    LOG_DBG("CEncConsumer release.\r\n");
    for (NvSciBufObj bufObj : m_pSciBufObjs) {
        if (bufObj != nullptr) {
            NvMediaIEPUnregisterNvSciBufObj(m_pNvMIEP.get(), bufObj);
        }
    }
    if(_enc_server_fd>0){
        close(_enc_server_fd);
        unlink(_socket_path);
    }

    UnregisterSyncObjs();

    if (m_encodeType == HW_VIDEO_REGDATACB_TYPE_AVC && m_stEncodeConfigH264Params.h264VUIParameters) {
        free(m_stEncodeConfigH264Params.h264VUIParameters);
        m_stEncodeConfigH264Params.h264VUIParameters = nullptr;
    }
    if (m_encodeType == HW_VIDEO_REGDATACB_TYPE_HEVC && m_stEncodeConfigH265Params.h265VUIParameters) {
        free(m_stEncodeConfigH265Params.h265VUIParameters);
        m_stEncodeConfigH265Params.h265VUIParameters = nullptr;
    }

    if (m_pOutputFile != nullptr) {
        fflush(m_pOutputFile);
        fclose(m_pOutputFile);
    }
}

// Buffer setup functions
SIPLStatus CEncConsumer::SetDataBufAttrList(NvSciBufAttrList &bufAttrList)
{
    auto status = NvMediaIEPFillNvSciBufAttrList(NVMEDIA_ENCODER_INSTANCE_0, bufAttrList);
    PCHK_NVMSTATUS_AND_RETURN(status, "NvMediaIEPFillNvSciBufAttrList");

    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType bufType = NvSciBufType_Image;
    bool needCpuAccess = false;
    bool isEnableCpuCache = false;

    /* Set all key-value pairs */
    NvSciBufAttrKeyValuePair attributes[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &isEnableCpuCache, sizeof(isEnableCpuCache) }
    };

    auto sciErr =
        NvSciBufAttrListSetAttrs(bufAttrList, attributes, sizeof(attributes) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

// Sync object setup functions
SIPLStatus CEncConsumer::SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList)
{
    if (m_pNvMIEP.get()) {
        auto nvmStatus = NvMediaIEPFillNvSciSyncAttrList(m_pNvMIEP.get(), signalerAttrList, NVMEDIA_SIGNALER);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Signaler NvMediaIEPFillNvSciSyncAttrList");

        nvmStatus = NvMediaIEPFillNvSciSyncAttrList(m_pNvMIEP.get(), waiterAttrList, NVMEDIA_WAITER);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Waiter NvMediaIEPFillNvSciSyncAttrList");
    } else {
        return NVSIPL_STATUS_NOT_INITIALIZED;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    if (m_pNvMIEP.get()) {
        PLOG_DBG("%s:Mapping data buffer, packetIndex: %u.\n", m_name.c_str(), packetIndex);
        //m_pSciBufObjs[packetIndex] = bufObj;
	NvSciBufObjDup( bufObj, &m_pSciBufObjs[packetIndex] );
        NvMediaStatus nvmStatus = NvMediaIEPRegisterNvSciBufObj(m_pNvMIEP.get(), m_pSciBufObjs[packetIndex]);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciBufObj");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::RegisterSignalSyncObj(NvSciSyncObj signalSyncObj)
{
    if (m_pNvMIEP.get()) {
        m_IEPSignalSyncObj = signalSyncObj;
        auto nvmStatus = NvMediaIEPRegisterNvSciSyncObj(m_pNvMIEP.get(), NVMEDIA_EOFSYNCOBJ, signalSyncObj);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciSyncObj for EOF");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj)
{
    if (m_pNvMIEP.get()) {
        auto nvmStatus = NvMediaIEPRegisterNvSciSyncObj(m_pNvMIEP.get(), NVMEDIA_PRESYNCOBJ, waiterSyncObj);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciSyncObj for PRE");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::UnregisterSyncObjs(void)
{
    PCHK_PTR_AND_RETURN(m_pNvMIEP.get(), "m_pNvMIEP nullptr");

    auto nvmStatus = NvMediaIEPUnregisterNvSciSyncObj(m_pNvMIEP.get(), m_IEPSignalSyncObj);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPUnregisterNvSciSyncObj for EOF");

    for (uint32_t i = 0U; i < m_numWaitSyncObj; ++i) {
        for (uint32_t j = 0U; j < m_elemsInfo.size(); ++j) {
            if (m_waiterSyncObjs[i][j]) {
                nvmStatus = NvMediaIEPUnregisterNvSciSyncObj(m_pNvMIEP.get(), m_waiterSyncObjs[i][j]);
            PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPUnregisterNvSciSyncObj for PRE");
            }
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{
    if (m_pNvMIEP.get()) {
        auto nvmStatus = NvMediaIEPInsertPreNvSciSyncFence(m_pNvMIEP.get(), &prefence);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPInsertPreNvSciSyncFence");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::SetEofSyncObj(void)
{
    if (m_pNvMIEP.get()) {
        auto nvmStatus = NvMediaIEPSetNvSciSyncObjforEOF(m_pNvMIEP.get(), m_IEPSignalSyncObj);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPSetNvSciSyncObjforEOF");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::EncodeOneFrame(NvSciBufObj pSciBufObj, u32 i_packetindex, uint8_t **ppOutputBuffer, size_t *pNumBytes, NvSciSyncFence *pPostfence) {
    NvMediaEncodePicParamsH264 encodePicParams;
    uint32_t uNumBytes = 0U;
    uint32_t uNumBytesAvailable = 0U;
    uint8_t *pBuffer = nullptr;
    struct hw_video_bufferinfo_t datacbinfo;

    if (_pvicconsumer == nullptr) {
        datacbinfo.timeinfo.framecapturetsc = static_cast<MetaData *>(m_metaPtrs[i_packetindex])->frameCaptureTSC * 32;
        datacbinfo.timeinfo.framecapturestarttsc = static_cast<MetaData *>(m_metaPtrs[i_packetindex])->frameCaptureStartTSC * 32;
        datacbinfo.timeinfo.exposurestarttime = (static_cast<MetaData *>(m_metaPtrs[i_packetindex])->frameCaptureStartTSC * 32)
        - (uint64_t)((static_cast<MetaData *>(m_metaPtrs[i_packetindex])->exposureTime[0]) * 1000000000)
        - (uint64_t)((static_cast<MetaData *>(m_metaPtrs[i_packetindex])->exposureTime[2]) * 1000000000);
        // at least 68 years back to 0
    }
    else {
        datacbinfo.timeinfo.framecapturetsc = ((CVICConsumer *)_pvicconsumer)->GetCurrentFrameCaptureTSC();
        datacbinfo.timeinfo.framecapturestarttsc = ((CVICConsumer *)_pvicconsumer)->GetCurrentFrameCaptureStartTSC();
        datacbinfo.timeinfo.exposurestarttime = ((CVICConsumer *)_pvicconsumer)->GetCurrentExposureStartTime();
        // at least 68 years back to 0
    }
    if (datacbinfo.timeinfo.framecapturetsc == 0) {
        // if tsc 0 then skip it
        return NVSIPL_STATUS_OK;
    }

    //set one frame params, default = 0
    memset(&encodePicParams, 0, sizeof(NvMediaEncodePicParamsH264));
    //IPP mode
    encodePicParams.pictureType = NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
    encodePicParams.encodePicFlags = NVMEDIA_ENCODE_PIC_FLAG_OUTPUT_SPSPPS;
    encodePicParams.nextBFrames    = 0;
    auto nvmStatus = NvMediaIEPFeedFrame(m_pNvMIEP.get(),             // *encoder
                                         pSciBufObj,                  // *frame
                                         &encodePicParams,            // encoder parameter
                                         NVMEDIA_ENCODER_INSTANCE_0); // encoder instance
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPFeedFrame");

    nvmStatus = NvMediaIEPGetEOFNvSciSyncFence(m_pNvMIEP.get(), m_IEPSignalSyncObj, pPostfence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, ": NvMediaIEPGetEOFNvSciSyncFence");

    bool bEncodeFrameDone = false;
    while(!bEncodeFrameDone) {
        NvMediaBitstreamBuffer bitstreams = {0};
        uNumBytesAvailable = 0U;
        uNumBytes = 0U;
        nvmStatus = NvMediaIEPBitsAvailable(m_pNvMIEP.get(),
                                            &uNumBytesAvailable,
                                            NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING,
                                            NVMEDIA_ENCODE_TIMEOUT_INFINITE);
        switch(nvmStatus) {
            case NVMEDIA_STATUS_OK:
                pBuffer = new (std::nothrow) uint8_t[uNumBytesAvailable];
                if (pBuffer == nullptr) {
                    PLOG_ERR("%s:Out of memory\r\n", m_name.c_str());
                    return NVSIPL_STATUS_OUT_OF_MEMORY;
                }

                if(_shmaddr){
                        bitstreams.bitstream = (uint8_t*)_shmaddr+SHM_HEAD_SIZE;
                        bitstreams.bitstreamSize = uNumBytesAvailable;
                        /* std::fill(_shmaddr, _shmaddr + uNumBytesAvailable, 0xE5); */
                        memset(_shmaddr,0xE5,uNumBytesAvailable);
                        //printf("use shm buf\n");
                }else{
                    bitstreams.bitstream = pBuffer;
                    bitstreams.bitstreamSize = uNumBytesAvailable;
                    std::fill(pBuffer, pBuffer + uNumBytesAvailable, 0xE5);
                }
                nvmStatus = NvMediaIEPGetBits(m_pNvMIEP.get(),
                                             &uNumBytes,
                                             1U,
                                             &bitstreams,
                                             nullptr);
                if(nvmStatus != NVMEDIA_STATUS_OK && nvmStatus != NVMEDIA_STATUS_NONE_PENDING) {
                    PLOG_ERR("%s:Error getting encoded bits\r\n", m_name.c_str());
                    free(pBuffer);
                    return NVSIPL_STATUS_ERROR;
                }

                if(uNumBytes != uNumBytesAvailable) {
                    PLOG_ERR("%s:Error-byte counts do not match %d vs. %d\r\n", m_name.c_str(),
                            uNumBytesAvailable, uNumBytes);
                    free(pBuffer);
                    return NVSIPL_STATUS_ERROR;
                }
                *ppOutputBuffer = pBuffer;
                *pNumBytes = (size_t)uNumBytesAvailable;
                bEncodeFrameDone = 1;
                break;

            case NVMEDIA_STATUS_PENDING:
                PLOG_ERR("%s:Error - encoded data is pending\r\n", m_name.c_str());
                return NVSIPL_STATUS_ERROR;

            case NVMEDIA_STATUS_NONE_PENDING:
                PLOG_ERR("%s:Error - no encoded data is pending\r\n", m_name.c_str());
                return NVSIPL_STATUS_ERROR;

            default:
                PLOG_ERR("%s:Error occured\r\n", m_name.c_str());
                return NVSIPL_STATUS_ERROR;
        }
    }

    if (HW_UNLIKELY(_datacb == nullptr)) {
        if (HW_UNLIKELY(_blogdatacbnull == 0)) {
            LOG_INFO("datacb is null!\r\n");
            _blogdatacbnull = 1;
        }
    }
    else {
        /*
        * We call the user data cb.
        */
        datacbinfo.regdatacbtype = _regdatacbtype;
        datacbinfo.blockindex = _blockindex;
        datacbinfo.sensorindex = _sensorindex;
        datacbinfo.outputtype = (HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE)_outputtype;
        datacbinfo.format_maintype = _usermaintype;
        datacbinfo.format_subtype = _usersubtype;
        if(_shmaddr){
            datacbinfo.pbuff = _shmaddr;
        }else{
            datacbinfo.pbuff = pBuffer;
        }
        datacbinfo.width = m_encodeWidth;
        datacbinfo.height = m_encodeHeight;
        datacbinfo.size = (size_t)uNumBytesAvailable;
        datacbinfo.bsynccb = 1;
        datacbinfo.bneedfree = 0;
        datacbinfo.pcustom = _pcontext;
        /*
        * Call the user data callback here.
        */
        _datacb(&datacbinfo);
    }
    if(_shmaddr){
        hw_video_shmimageheader_t* header = reinterpret_cast<hw_video_shmimageheader_t*>(_shmaddr);
        header->magic_number = m_encodeType;
        header->width = m_encodeWidth;
        header->height = m_encodeHeight;
        // Convert NVIDIA internal timestamp to milliseconds
        // nv_timestamp: NVIDIA internal timestamp value
        // ms: Converted timestamp in milliseconds
        // Formula: nv_timestamp * 32 / 1000000 = ms
        // Here, we are using nanoseconds as the unit

        // Code Example:
        // uint64_t nv_timestamp = m_metaPtrs[packetIndex]->frameCaptureTSC; // NVIDIA internal timestamp
        // uint64_t ms = (nv_timestamp * 32) / 1 000 000;
        header->timestamp = datacbinfo.timeinfo.exposurestarttime;
    }

    std::lock_guard<std::mutex> lock(_rwlock);
    char command[128];
    /* char command[] = "bufready"; */
    sprintf(command,"bufready:%d",uNumBytesAvailable);
    _clientready=0;
    for (std::vector<ElementInfo>::size_type i = 0; i < _client.size(); i++) {
        int fd = _client[i];

        int write_len = send(fd, command, strlen(command),0);
        if (write_len <= 0) {
            continue;
        }else{
            _clientready++;
        }
    }

    return NVSIPL_STATUS_OK;
}

// Streaming functions
SIPLStatus CEncConsumer::ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence)
{
    PLOG_DBG("%s:Process payload (packetIndex = 0x%x).\r\n", m_name.c_str(), packetIndex);
    {
        std::lock_guard<std::mutex> lock(_rwlock);
        if(_clientready>0 || (_blockindex == 2)){//shm buf has not been consumed by all clients
            /* if(_clientready>0 || (_skipfreamCount++%2)== 0){//shm buf has not been consumed by all clients */
            /* _skipfreamCount %= 2; */
            /* printf("skip frame +++++++++ %d\n",_skipfreamCount); */
            return NVSIPL_STATUS_OK;//skip this frame
        }
    }

    auto status = EncodeOneFrame(m_pSciBufObjs[packetIndex], packetIndex, & m_pEncodedBuf, &m_encodedBytes, pPostfence);
    PCHK_STATUS_AND_RETURN(status, "ProcessPayload");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::OnProcessPayloadDone(uint32_t packetIndex)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    //dump frames to local file
    if (m_consConfig.bFileDump && (m_frameNum >= DUMP_START_FRAME && m_frameNum <= DUMP_END_FRAME)) {
        if (m_pEncodedBuf && m_encodedBytes > 0) {
            if(fwrite(m_pEncodedBuf, m_encodedBytes, 1, m_pOutputFile) != 1) {
                PLOG_ERR("%s:Error writing %d bytes\r\n", m_name.c_str(), m_encodedBytes);
                status = NVSIPL_STATUS_ERROR;
                goto cleanup;
            }
            PLOG_DBG("%s:writing %u bytes, m_frameNum %u\r\n", m_name.c_str(), m_encodedBytes, m_frameNum);
            fflush(m_pOutputFile);
        }
    }
    PLOG_DBG("%s:ProcessPayload succ.\r\n", m_name.c_str());

cleanup:
    if (m_pEncodedBuf) {
        delete[] m_pEncodedBuf;
        m_pEncodedBuf = nullptr;
    }
    m_encodedBytes = 0;

    return status;
}

SIPLStatus CEncConsumer::OnDataBufAttrListReceived(NvSciBufAttrList bufAttrList)
{
    if (nullptr == m_pNvMIEP.get()) {
    auto status = InitEncoder(bufAttrList);
    PCHK_STATUS_AND_RETURN(status, "InitEncoder");
    }

    return NVSIPL_STATUS_OK;
}

void CEncConsumer::ProcessEventHandle(CEncConsumer* context)
{
    int client_fd;
    struct sockaddr_un client_addr;
    socklen_t client_len = sizeof(client_addr);

    if(context->_enc_server_fd<=0){
        return;
    }
    if (listen(context->_enc_server_fd, MAX_CLIENTS) < 0) {
        PLOG_DBG("listen error\n");
        return;
    }

    while(true){
        client_fd = accept(context->_enc_server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            PLOG_DBG("accept error\n");
            continue;
        }
        std::lock_guard<std::mutex> lock(context->_rwlock);
        context->_client.push_back(client_fd);
        std::thread clientEventThread(ProcessClientEventHandle,context,client_fd);
        clientEventThread.detach();
        /* context->_alivecount++; */


    }

}

void CEncConsumer::ProcessClientEventHandle(CEncConsumer* context,int clientfd)
{
    char command[128];
    int read_len = -1;

    while(true){
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(clientfd, &readfds);
        struct timeval timeout;
        timeout.tv_sec = 2;  // set timeout 2s.
        timeout.tv_usec = 0;
        int ret = select(clientfd + 1, &readfds, NULL, NULL, &timeout);
        if(ret==0){//timeout
            continue;
        }
        if(ret<0){//exit
            break;
        }
        std::lock_guard<std::mutex> lock(context->_rwlock);
        if(FD_ISSET(clientfd,&readfds)){
            read_len = recv(clientfd, command, sizeof(command), MSG_PEEK | MSG_DONTWAIT);
            if (read_len <= 0) {
                LOG_DBG("enc client[%d] disconnect.\r\n",clientfd);
                break;
            }
            memset(command,0,128);
            read_len = recv(clientfd, command, sizeof(command), 0);
            if(read_len<0){
                continue;
            }
            command[read_len] = '\0';
            if (strcmp(command, "unlock") == 0) {
                // exec attach
                context->_clientready--;
            } else {
                // unknow command
            }

        }
    }

    std::lock_guard<std::mutex> lock(context->_rwlock);
    for (std::vector<int>::iterator it = context->_client.begin(); it != context->_client.end(); ++it) {
        if (*it == clientfd) {
            context->_client.erase(it);
            break;
        }
    }
    close(clientfd);
}

int CEncConsumer::getShmKeyBySensorID(int blockidx,int sensoridx){
    int index = (blockidx<<4) | sensoridx;
    switch(index){
        case 0x00:
            return SHM_KEY_BLOCK0_SENSOR0;
        case 0x01:
            return SHM_KEY_BLOCK0_SENSOR1;
        case 0x02:
            return SHM_KEY_BLOCK0_SENSOR2;
        case 0x03:
            return SHM_KEY_BLOCK0_SENSOR3;
        case 0x10:
            return SHM_KEY_BLOCK1_SENSOR0;
        case 0x11:
            return SHM_KEY_BLOCK1_SENSOR1;
        case 0x12:
            return SHM_KEY_BLOCK1_SENSOR2;
        case 0x13:
            return SHM_KEY_BLOCK1_SENSOR3;
        case 0x20:
            return SHM_KEY_BLOCK2_SENSOR0;
        case 0x21:
            return SHM_KEY_BLOCK2_SENSOR1;
        case 0x22:
            return SHM_KEY_BLOCK2_SENSOR2;
        case 0x23:
            return SHM_KEY_BLOCK2_SENSOR3;
        default:
            return -1;
    }
}
