// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CEncConsumer.hpp"
#include "nvscibuf.h"
#include "CEncManager.h"

CEncConsumer::CEncConsumer(NvSciStreamBlock handle, uint32_t uSensor, NvSciStreamBlock queueHandle)
    : CConsumer("EncConsumer", handle, uSensor, queueHandle)
{
}

SIPLStatus CEncConsumer::HandleClientInit(void)
{
    if (m_pAppConfig->IsFileDumped()) {
        string fileName = "multicast_enc" + to_string(m_uSensorId) + ".h264";
        m_pOutputFile = fopen(fileName.c_str(), "wb");
        PCHK_PTR_AND_RETURN(m_pOutputFile, "Open encoder output file");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::SetEncodeConfig(void)
{
    memset(&m_stEncodeConfigH264Params, 0, sizeof(NvMediaEncodeConfigH264));
    m_stEncodeConfigH264Params.h264VUIParameters =
        (NvMediaEncodeConfigH264VUIParams *)calloc(1, sizeof(NvMediaEncodeConfigH264VUIParams));
    CHK_PTR_AND_RETURN(m_stEncodeConfigH264Params.h264VUIParameters, "Alloc h264VUIParameters failed");
    m_stEncodeConfigH264Params.h264VUIParameters->timingInfoPresentFlag = 1;

    // Setting Up Config Params
    m_stEncodeConfigH264Params.features = NVMEDIA_ENCODE_CONFIG_H264_ENABLE_OUTPUT_AUD;
    m_stEncodeConfigH264Params.gopLength = 30;
    m_stEncodeConfigH264Params.idrPeriod = 300;
    m_stEncodeConfigH264Params.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_INTRA_FRAMES;
    m_stEncodeConfigH264Params.adaptiveTransformMode = NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_AUTOSELECT;
    m_stEncodeConfigH264Params.bdirectMode = NVMEDIA_ENCODE_H264_BDIRECT_MODE_DISABLE;
    m_stEncodeConfigH264Params.entropyCodingMode = NVMEDIA_ENCODE_H264_ENTROPY_CODING_MODE_CAVLC;
    m_stEncodeConfigH264Params.encPreset = NVMEDIA_ENC_PRESET_HP;

    m_stEncodeConfigH264Params.rcParams.rateControlMode = NVMEDIA_ENCODE_PARAMS_RC_CONSTQP;
    m_stEncodeConfigH264Params.rcParams.params.const_qp.constQP.qpIntra = 32;
    m_stEncodeConfigH264Params.rcParams.params.const_qp.constQP.qpInterP = 35;
    m_stEncodeConfigH264Params.rcParams.params.const_qp.constQP.qpInterB = 25;
    m_stEncodeConfigH264Params.rcParams.numBFrames = 0;
    auto nvmediaStatus = NvMediaIEPSetConfiguration(m_pNvMIEP.get(), &m_stEncodeConfigH264Params);
    PCHK_NVMSTATUS_AND_RETURN(nvmediaStatus, "NvMediaIEPSetConfiguration failed");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::InitEncoder(NvSciBufAttrList bufAttrList)
{
    uint16_t encodeWidth = 0U;
    uint16_t encodeHeight = 0U;

    SIPLStatus status = NVSIPL_STATUS_ERROR;
    NvMediaEncodeInitializeParamsH264 encoderInitParams;
    memset(&encoderInitParams, 0, sizeof(encoderInitParams));
    encoderInitParams.profile = NVMEDIA_ENCODE_PROFILE_AUTOSELECT;
    encoderInitParams.level = NVMEDIA_ENCODE_LEVEL_AUTOSELECT;

    if (m_pAppConfig != nullptr) {
        status = m_pAppConfig->GetResolutionWidthAndHeight(m_uSensorId, encodeWidth, encodeHeight);
        PCHK_STATUS_AND_RETURN(status, "GetResolutionWidthAndHeight");
    } else {
        return NVSIPL_STATUS_NOT_INITIALIZED;
    }
    encoderInitParams.encodeHeight = encodeHeight;
    encoderInitParams.encodeWidth = encodeWidth;
    encoderInitParams.useBFramesAsRef = 0;
    encoderInitParams.frameRateDen = 1;
    encoderInitParams.frameRateNum = 30;
    encoderInitParams.maxNumRefFrames = 1;
    encoderInitParams.enableExternalMEHints = NVMEDIA_FALSE;
    encoderInitParams.enableAllIFrames = NVMEDIA_FALSE;

    m_pNvMIEP.reset(NvMediaIEPCreate(NVMEDIA_IMAGE_ENCODE_H264,    // codec
                                     &encoderInitParams,           // init params
                                     bufAttrList,                  // reconciled attr list
                                     0,                            // maxOutputBuffering
                                     NVMEDIA_ENCODER_INSTANCE_0)); // encoder instance
    PCHK_PTR_AND_RETURN(m_pNvMIEP, "NvMediaIEPCreate");

    status = SetEncodeConfig();
    return status;
}

CEncConsumer::~CEncConsumer(void)
{
    LOG_DBG("CEncConsumer release.\n");
    for (NvSciBufObj bufObj : m_pSciBufObjs) {
        if (bufObj != nullptr) {
            NvMediaIEPUnregisterNvSciBufObj(m_pNvMIEP.get(), bufObj);
        }
    }

    UnregisterSyncObjs();

    if (m_stEncodeConfigH264Params.h264VUIParameters) {
        free(m_stEncodeConfigH264Params.h264VUIParameters);
        m_stEncodeConfigH264Params.h264VUIParameters = nullptr;
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
        PLOG_DBG("Mapping data buffer, packetIndex: %u.\n", packetIndex);
        m_pSciBufObjs[packetIndex] = bufObj;
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

SIPLStatus CEncConsumer::EncodeOneFrame(NvSciBufObj pSciBufObj,
                                        uint8_t **ppOutputBuffer,
                                        size_t *pNumBytes,
                                        NvSciSyncFence *pPostfence)
{
    NvMediaEncodePicParamsH264 encodePicParams;
    uint32_t uNumBytes = 0U;
    uint32_t uNumBytesAvailable = 0U;
    uint8_t *pBuffer = nullptr;

    //set one frame params, default = 0
    memset(&encodePicParams, 0, sizeof(NvMediaEncodePicParamsH264));
    //IPP mode
    encodePicParams.pictureType = (m_frameNum == DUMP_START_FRAME) ? 
                                  NVMEDIA_ENCODE_PIC_TYPE_IDR : NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
    encodePicParams.encodePicFlags = NVMEDIA_ENCODE_PIC_FLAG_OUTPUT_SPSPPS;
    encodePicParams.nextBFrames = 0;
    auto nvmStatus = NvMediaIEPFeedFrame(m_pNvMIEP.get(),             // *encoder
                                         pSciBufObj,                  // *frame
                                         &encodePicParams,            // encoder parameter
                                         NVMEDIA_ENCODER_INSTANCE_0); // encoder instance
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPFeedFrame");

    nvmStatus = NvMediaIEPGetEOFNvSciSyncFence(m_pNvMIEP.get(), m_IEPSignalSyncObj, pPostfence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, ": NvMediaIEPGetEOFNvSciSyncFence");

    bool bEncodeFrameDone = false;
    while (!bEncodeFrameDone) {
        NvMediaBitstreamBuffer bitstreams = { 0 };
        uNumBytesAvailable = 0U;
        uNumBytes = 0U;
        nvmStatus = NvMediaIEPBitsAvailable(m_pNvMIEP.get(), &uNumBytesAvailable,
                                            NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING, NVMEDIA_ENCODE_TIMEOUT_INFINITE);
        switch (nvmStatus) {
            case NVMEDIA_STATUS_OK:
                pBuffer = new (std::nothrow) uint8_t[uNumBytesAvailable];
                if (pBuffer == nullptr) {
                    PLOG_ERR("Out of memory\n");
                    return NVSIPL_STATUS_OUT_OF_MEMORY;
                }

                bitstreams.bitstream = pBuffer;
                bitstreams.bitstreamSize = uNumBytesAvailable;
                std::fill(pBuffer, pBuffer + uNumBytesAvailable, 0xE5);
                nvmStatus = NvMediaIEPGetBits(m_pNvMIEP.get(), &uNumBytes, 1U, &bitstreams, nullptr);
                if (nvmStatus != NVMEDIA_STATUS_OK && nvmStatus != NVMEDIA_STATUS_NONE_PENDING) {
                    PLOG_ERR("Error getting encoded bits\n");
                    free(pBuffer);
                    return NVSIPL_STATUS_ERROR;
                }

                if (uNumBytes != uNumBytesAvailable) {
                    PLOG_ERR("Error-byte counts do not match %d vs. %d\n", uNumBytesAvailable, uNumBytes);
                    free(pBuffer);
                    return NVSIPL_STATUS_ERROR;
                }
                *ppOutputBuffer = pBuffer;
                *pNumBytes = (size_t)uNumBytesAvailable;
                bEncodeFrameDone = 1;
                break;

            case NVMEDIA_STATUS_PENDING:
                PLOG_ERR("Error - encoded data is pending\n");
                return NVSIPL_STATUS_ERROR;

            case NVMEDIA_STATUS_NONE_PENDING:
                PLOG_ERR("Error - no encoded data is pending\n");
                return NVSIPL_STATUS_ERROR;

            default:
                PLOG_ERR("Error occured\n");
                return NVSIPL_STATUS_ERROR;
        }
    }

    return NVSIPL_STATUS_OK;
}

// Streaming functions
SIPLStatus CEncConsumer::ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence)
{
    PLOG_DBG("Process payload (packetIndex = 0x%x).\n", packetIndex);

    auto status = EncodeOneFrame(m_pSciBufObjs[packetIndex], &m_pEncodedBuf, &m_encodedBytes, pPostfence);
    PCHK_STATUS_AND_RETURN(status, "ProcessPayload");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CEncConsumer::OnProcessPayloadDone(uint32_t packetIndex)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    //dump frames to local file
    if (m_pAppConfig->IsFileDumped() && (m_frameNum >= DUMP_START_FRAME && m_frameNum <= DUMP_END_FRAME)) {
        if (m_pEncodedBuf && m_encodedBytes > 0) {
            if (fwrite(m_pEncodedBuf, m_encodedBytes, 1, m_pOutputFile) != 1) {
                PLOG_ERR("Error writing %d bytes\n", m_encodedBytes);
                status = NVSIPL_STATUS_ERROR;
                goto cleanup;
            }
            PLOG_DBG("writing %u bytes, m_frameNum %u\n", m_encodedBytes, m_frameNum);
            fflush(m_pOutputFile);
        }
    }

    {
        hozon::netaos::multicast::Multicast_EncodedImage encoded_image;
        encoded_image.data.assign(reinterpret_cast<char*>(m_pEncodedBuf), m_encodedBytes);
        CEncManager::Instance().NotifyEncodedImage(m_uSensorId, encoded_image);
    }
    PLOG_DBG("ProcessPayload succ.\n");

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
