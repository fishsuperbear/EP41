#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include "nv_camera_impl.hpp"

using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace camera {

// Process events from the notification queue.
void ThreadData::process(const NvSIPLPipelineNotifier::NotificationData& event) {
    if (event.eNotifType < NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DROP) {
        // Don't print information events, these completion notifications are redundant since
        // the frame sequence counter is already being printed
        return;
    } else if (event.eNotifType == NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DROP) {
        uNumFrameDrops++;
    } else if (event.eNotifType == NvSIPLPipelineNotifier::NotificationType::NOTIF_WARN_ICP_FRAME_DISCONTINUITY) {
        uNumFrameDiscontinuities++;
    }

    const char *eventName = nullptr;
    const SIPLStatus status = GetEventName(event, eventName);
    if ((status != NVSIPL_STATUS_OK) || (eventName == nullptr)) {
        CAM_LOG_ERROR << "Failed to get event name.";
        return;
    } else {
        printMutex->lock();
        CAM_LOG_TRACE << threadName << ": " << eventName;
        printMutex->unlock();
    }
}

// Process completed images.
void ThreadData::process(INvSIPLClient::INvSIPLBuffer* const & pBuffer) {
    INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer =
        static_cast<INvSIPLClient::INvSIPLNvMBuffer *>(pBuffer);
    if (pNvMBuffer == nullptr) {
        CAM_LOG_ERROR << "Invalid buffer.";
    } else {
        if (uSensorId >= MAX_NUM_SENSORS) {
            CAM_LOG_ERROR << "Invalid sensor index : " <<  uSensorId;
            std::terminate();
        }
        const INvSIPLClient::ImageMetaData& metadata = pNvMBuffer->GetImageData();
        uint32_t frame_counter = (metadata.frameSeqNumInfo.frameSeqNumValid ?
                                    metadata.frameSeqNumInfo.frameSequenceNumber : uFrameCounter);
        printMutex->lock();
        CAM_LOG_TRACE << threadName << ": "<< frame_counter;
        printMutex->unlock();

        image_info.time_stamp = GetRealTimestamp();
        image_info.frame_id  = frame_counter;

        nvbufptr = pNvMBuffer;

        if (on_data_frame) {
            on_data_frame();
        }

        if (nvstream_post) {
            nvstream_post(pNvMBuffer);
        }

        nvbufptr = nullptr;

        CAM_LOG_DEBUG << "Camera process receive. ";
        // Write Image File
        // WriteImageToFile(pNvMBuffer, uSensorId, frame_counter);

        const SIPLStatus status = pNvMBuffer->Release();
        if (status != NVSIPL_STATUS_OK) {
            CAM_LOG_ERROR << "Buffer release failed.";
        }

        uFrameCounter++;
    }
}

SIPLStatus ThreadData::GetSIPLBuffer(std::string& image_data, uint32_t sensor_id) {
    NvSciError sciErr = NvSciError_Success;
    SIPLStatus status = NVSIPL_STATUS_OK;

    INvSIPLClient::INvSIPLNvMBuffer * pNvmBuf = nvbufptr;
    CFileWriter file_writer;

    if (m_cpuWaitContext != nullptr) {
        // Wait on EOF fence if its not ICP
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        status = pNvmBuf->GetEOFNvSciSyncFence(&fence);
        CHK_STATUS_AND_RETURN(status,
                "INvSIPLClient::INvSIPLNvMBuffer::GetEOFNvSciSyncFence");

        sciErr = NvSciSyncFenceWait(&fence, m_cpuWaitContext, FENCE_FRAME_TIMEOUT_MS * 1000UL);
        if (sciErr != NvSciError_Success) {
            if (sciErr == NvSciError_Timeout) {
                CAM_LOG_ERROR << "Frame done NvSciSyncFenceWait timed out";
            } else {
                CAM_LOG_ERROR << "Frame done NvSciSyncFenceWait failed";
            }
        }
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait Failed");
        NvSciSyncFenceClear(&fence);
    }

    // Write Buffer
    NvSciBufObj bufPtr = pNvmBuf->GetNvSciBufImage();
    BufferAttrs bufAttrs;
    status = PopulateBufAttr(bufPtr, bufAttrs);
    if(status != NVSIPL_STATUS_OK) {
        CAM_LOG_ERROR << "FileWriter: PopulateBufAttr failed";
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    uint32_t numSurfaces = -1U;
    float *xScalePtr = nullptr, *yScalePtr = nullptr;
    uint32_t *bytesPerPixelPtr = nullptr;
    bool isPackedYUV = false;
    status = file_writer.GetBuffParams(bufAttrs,
                            &xScalePtr,
                            &yScalePtr,
                            &bytesPerPixelPtr,
                            &numSurfaces,
                            &isPackedYUV);
    if (status != NVSIPL_STATUS_OK) {
        CAM_LOG_ERROR << "GetBuffParams failed";
        return status;
    }

    if (isPackedYUV)
    {
        NvSciError sciErr;

        void* va_ptr = nullptr;
        sciErr = NvSciBufObjGetConstCpuPtr(bufPtr, (const void**)&va_ptr);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetConstCpuPtr Failed");

        uint32_t image_size  = bufAttrs.size;
        image_info.height    = bufAttrs.planeHeights[0];
        image_info.width     = bufAttrs.planeWidths[0];

        std::string image_yuyv;
        image_yuyv.resize(image_size, '\0');
        uint8_t* basePtr = static_cast<uint8_t*>(va_ptr);

        for (uint32_t j = 0U; j < bufAttrs.planeHeights[0]; j++) {
            memcpy((reinterpret_cast<uint8_t*>(&image_yuyv[0]) + j * (bufAttrs.planePitches[0])),
                basePtr + j * bufAttrs.planePitches[0], bufAttrs.planeWidths[0] * bytesPerPixelPtr[0]);
        }

        YUYV2NV12(image_info.width, image_info.height, image_yuyv, image_data);
    } else {
        uint32_t pBuffPitches[MAX_NUM_SURFACES] = { 0U };
        uint8_t *pBuff[MAX_NUM_SURFACES] = { 0U };
        uint32_t size[MAX_NUM_SURFACES] = { 0U };
        uint32_t imageSize = 0U;

        uint32_t height = bufAttrs.planeHeights[0];
        uint32_t width = bufAttrs.planeWidths[0];
        for (uint32_t i = 0U; i < numSurfaces; i++) {
            size[i] = (width * xScalePtr[i] * height * yScalePtr[i] * bytesPerPixelPtr[i]);
            imageSize += size[i];
            pBuffPitches[i] = (uint32_t)((float)width * xScalePtr[i]) * bytesPerPixelPtr[i];
        }

        std::string image_yuv420;
        if (image_yuv420.size() == 0) {
            image_yuv420.resize(imageSize, '\0');
        }

        uint8_t *buffIter = reinterpret_cast<uint8_t*>(&image_yuv420[0]);
        for (uint32_t i = 0U; i < numSurfaces; i++) {
            pBuff[i] = buffIter;
            buffIter += (uint32_t)(height * yScalePtr[i] * pBuffPitches[i]);
        }

        sciErr = NvSciBufObjGetPixels(bufPtr, nullptr, (void **)pBuff, size, pBuffPitches);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetPixels Failed");

        YUY4202NV12(width, height, image_yuv420, image_data);
    }
    return NVSIPL_STATUS_OK;
}


NvCameraImpl::NvCameraImpl() {

}

NvCameraImpl::~NvCameraImpl() {

}

SIPLStatus NvCameraImpl::GetEEPROMData(INvSIPLCamera *siplCamera, uint32_t sensorid)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    CameraInternalData camera_interal_data;
    uint16_t addr = 0x0;
    uint32_t length = 0;

    if (sensorid <= 7) {
        addr = 0x0;
        length = 0xff;
    } else { // to do for isx031
        return status;
    }

    camera_interal_data.moduleInfo = to_string(sensorid);
    camera_interal_data.data.resize(length);

    status = siplCamera->ReadEEPROMData(sensorid, addr, length, camera_interal_data.data.data());
    if (status != NVSIPL_STATUS_OK) {
        CAM_LOG_ERROR << "Failed to read EEPROM data via SIPL API";
        return status;
    }

    // test print
    camera_interal_data_x3f_x8b *data = 
            reinterpret_cast<camera_interal_data_x3f_x8b*>(camera_interal_data.data.data());
    CAM_LOG_INFO << "Camera params "
            << " version : " << data->version
            << " K1 : " << data->k1
            << " K2 : " << data->k2
            << " K3 : " << data->k3
            << " p1 : " << data->p1
            << " p2 : " << data->p2;

    return status;
}

SIPLStatus NvCameraImpl::Init(SensorConfig &config) {
    _config = config;
    SIPLStatus status = NVSIPL_STATUS_OK;
    // SIPL Log level
    INvSIPLQueryTrace::GetInstance()->SetLevel(INvSIPLQueryTrace::LevelNone);
    INvSIPLTrace::GetInstance()->SetLevel(INvSIPLTrace::LevelNone);

    // INvSIPLQuery
    auto pQuery = INvSIPLQuery::GetInstance();
    CHK_PTR_AND_RETURN(pQuery, "INvSIPLQuery::GetInstance");

    status = pQuery->ParseDatabase();
    CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::ParseDatabase");

    CAM_LOG_INFO << "Getting platform configuration for : " << config.platform_name;
    status = pQuery->GetPlatformCfg(config.platform_name, oPlatformCfg);
    CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::GetPlatformCfg");

    // Apply mask
    if (config.vMasks.size() != 0) {
        for(auto it : config.vMasks) {
            CAM_LOG_INFO << "Setting link masks : " << it;
        }
        status = pQuery->ApplyMask(oPlatformCfg, config.vMasks);
        CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::ApplyMask");
    }

    siplCamera = INvSIPLCamera::GetInstance();
    CHK_PTR_AND_RETURN(siplCamera, "INvSIPLCamera::GetInstance()");

    // SIPL Platform config
    status = siplCamera->SetPlatformCfg(&oPlatformCfg, deviceBlockQueues);
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::SetPlatformCfg()");

    // parse pipeline config
    std::unordered_map<std::string, NvSIPLPipelineConfiguration> config_pipeline;
    for (uint32_t i = 0; i < sizeof(config.pipeline)/sizeof(config.pipeline[0]); i++) {
        std::string sensor_name = config.pipeline[i].sensor_name;
        NvSIPLPipelineConfiguration pipelineCfg;
        pipelineCfg.captureOutputRequested = config.pipeline[i].captureOutputRequested;
        pipelineCfg.isp0OutputRequested = config.pipeline[i].isp0OutputRequested;
        pipelineCfg.isp1OutputRequested = config.pipeline[i].isp1OutputRequested;
        pipelineCfg.isp2OutputRequested = config.pipeline[i].isp2OutputRequested;
        config_pipeline.insert(std::pair<std::string, NvSIPLPipelineConfiguration>(sensor_name, pipelineCfg));
    }

    // set pipeline config for each sensor
    for (auto d = 0U; d != oPlatformCfg.numDeviceBlocks; d++) {
        auto db = oPlatformCfg.deviceBlockList[d];
        for (auto m = 0U; m != db.numCameraModules; m++) {
            auto mod = db.cameraModuleInfoList[m];
            auto sensor = mod.sensorInfo;
            if (config_pipeline.find(sensor.name) == config_pipeline.end()) {
                CAM_LOG_ERROR << "Not find sensor pipeline config. " << sensor.name;
                continue;
            }
            vSensorIds.push_back(sensor.id);
            vSensorInfos.push_back(sensor);
            status = siplCamera->SetPipelineCfg(sensor.id, config_pipeline[sensor.name], queues[sensor.id]);
            vPipelineInfos.insert(std::pair<uint32_t, NvSIPLPipelineConfiguration>(sensor.id, config_pipeline[sensor.name]));
            CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::SetPipelineCfg()");
        }
    }

    status = siplCamera->Init();

    for (const auto& uSensor : vSensorIds) {
        GetEEPROMData(siplCamera.get(), uSensor);
    }

    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Init()");

    return status;
}

SIPLStatus NvCameraImpl::ImageManagerInit() {
    auto status = imageManager.Init(siplCamera.get(), vPipelineInfos);
    CHK_STATUS_AND_RETURN(status, "CImageManager::Init()");

    for (const auto& uSensor : vSensorIds) {
        status = imageManager.Allocate(uSensor);
        CHK_STATUS_AND_RETURN(status, "CImageManager::Allocate()");
        status = imageManager.Register(uSensor);
        CHK_STATUS_AND_RETURN(status, "CImageManager::Register()");
    }

    return status;
}

INvSIPLCamera* NvCameraImpl::GetSIPLCamera() {
    return siplCamera.get();
}

std::unordered_map<uint32_t, NvSIPLPipelineConfiguration>& NvCameraImpl::GetSensorInfo() {
    return vPipelineInfos;
}

SIPLStatus NvCameraImpl::InitCameraNito() {
    SIPLStatus status = NVSIPL_STATUS_OK;

    for (const auto& uSensor : vSensorInfos) {
        std::vector<uint8_t> blob;
        if (uSensor.name == SENSOR_0X8B40) {
            if (uSensor.id == 0) {
                status = LoadNitoFile(NITO_PATH, X8B40_F120_NITO, blob, defaultNitoLoaded);
                CHK_STATUS_AND_RETURN(status, "LoadNitoFile()");
            } else if (uSensor.id == 1) {
                status = LoadNitoFile(NITO_PATH, X8B40_F30_NITO, blob, defaultNitoLoaded);
                CHK_STATUS_AND_RETURN(status, "LoadNitoFile()");
            }
        } else if (uSensor.name == SENSOR_0X03F) {
            status = LoadNitoFile(NITO_PATH, X3F_NITO, blob, defaultNitoLoaded);
            CHK_STATUS_AND_RETURN(status, "LoadNitoFile()");
        } else {
            CAM_LOG_TRACE << "Camera not support soc isp, sensor : " << uSensor.name;
            continue;
        }
        status = siplCamera->RegisterAutoControlPlugin(uSensor.id, NV_PLUGIN, nullptr, blob);
        CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterAutoControlPlugin()");
    }

    return status;
}

SIPLStatus NvCameraImpl::Start() {
    SIPLStatus status = NVSIPL_STATUS_OK;

    // Nito init before image buf register
    InitCameraNito();

    for (const auto& uSensor : vSensorIds) {
        for (uint32_t i = 0U; i < THREAD_INDEX_COUNT; i++) {
            threadDataStructs[uSensor][i].printMutex = &threadPrintMutex;
            threadDataStructs[uSensor][i].uSensorId = uSensor;
            switch (i) {
                case THREAD_INDEX_ICP:
                    threadDataStructs[uSensor][i].threadName = "ICP(Sensor:" + std::to_string(uSensor)+")";
                    threadDataStructs[uSensor][i].imageQueueHandler.Start(queues[uSensor].captureCompletionQueue,
                                                                 &threadDataStructs[uSensor][i],
                                                                 IMAGE_QUEUE_TIMEOUT_US);
                    break;
                case THREAD_INDEX_ISP0:
                    threadDataStructs[uSensor][i].threadName = "ISP0(Sensor:" + std::to_string(uSensor)+")";
                    threadDataStructs[uSensor][i].imageQueueHandler.Start(queues[uSensor].isp0CompletionQueue,
                                                                 &threadDataStructs[uSensor][i],
                                                                 IMAGE_QUEUE_TIMEOUT_US);
                    break;
                case THREAD_INDEX_ISP1:
                    threadDataStructs[uSensor][i].threadName = "ISP1(Sensor:" + std::to_string(uSensor)+")";
                    threadDataStructs[uSensor][i].imageQueueHandler.Start(queues[uSensor].isp1CompletionQueue,
                                                                 &threadDataStructs[uSensor][i],
                                                                 IMAGE_QUEUE_TIMEOUT_US);
                    break;
                case THREAD_INDEX_ISP2:
                    threadDataStructs[uSensor][i].threadName = "ISP2(Sensor:" + std::to_string(uSensor)+")";
                    threadDataStructs[uSensor][i].imageQueueHandler.Start(queues[uSensor].isp2CompletionQueue,
                                                                 &threadDataStructs[uSensor][i],
                                                                 IMAGE_QUEUE_TIMEOUT_US);
                    break;
                case THREAD_INDEX_EVENT:
                    threadDataStructs[uSensor][i].threadName = "Event(Sensor:" + std::to_string(uSensor)+")";
                    threadDataStructs[uSensor][i].eventQueueHandler.Start(queues[uSensor].notificationQueue,
                                                                 &threadDataStructs[uSensor][i],
                                                                 EVENT_QUEUE_TIMEOUT_US);
                    break;
                default:
                    CAM_LOG_ERROR << "Unexpected thread index.";
                    return NVSIPL_STATUS_ERROR;
            }
        }
    }

    status = siplCamera->Start();
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Start()");

    return status;
}

SIPLStatus NvCameraImpl::Deinit() {
    SIPLStatus status = NVSIPL_STATUS_OK;
    CAM_LOG_INFO << "Camera deinit start.";

    for (const auto& uSensor : vSensorIds) {
        for (uint32_t i = 0U; i < THREAD_INDEX_COUNT; i++) {
            if (threadDataStructs[uSensor][i].imageQueueHandler.IsRunning()) {
                threadDataStructs[uSensor][i].imageQueueHandler.Stop();
            }
            if (threadDataStructs[uSensor][i].eventQueueHandler.IsRunning()) {
                threadDataStructs[uSensor][i].eventQueueHandler.Stop();
            }
        }
        CAM_LOG_INFO << "Sensor" << uSensor
                  << "Frame drops: "
                  << threadDataStructs[uSensor][THREAD_INDEX_EVENT].uNumFrameDrops
                  << "Frame discontinuities: "
                  << threadDataStructs[uSensor][THREAD_INDEX_EVENT].uNumFrameDiscontinuities;
    }

    status = siplCamera->Stop();
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Stop()");

    status = siplCamera->Deinit();
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::Deinit()");

    CAM_LOG_INFO << "Camera deinit end.";
    return status;
}

void NvCameraImpl::ImageManagerDeInit() {
    imageManager.Deinit();
}

int32_t NvCameraImpl::RegisterNvStreamPost(NvStreamCallback callback, uint32_t sensor, uint32_t imageCaptureType) {
    threadDataStructs[sensor][imageCaptureType].nvstream_post = callback;
    return 0;
}

int32_t NvCameraImpl::GetCaptureType(uint32_t sensor) {
    if (vPipelineInfos.find(sensor) == vPipelineInfos.end()) {
        return -1;
    }

    if (vPipelineInfos[sensor].captureOutputRequested) {
        return THREAD_INDEX_ICP;
    } else if (vPipelineInfos[sensor].isp0OutputRequested) {
        return THREAD_INDEX_ISP0;
    } else if (vPipelineInfos[sensor].isp1OutputRequested) {
        return THREAD_INDEX_ISP1;
    } else if (vPipelineInfos[sensor].isp2OutputRequested) {
        return THREAD_INDEX_ISP2;
    }

    return THREAD_INDEX_ICP;
}

int32_t NvCameraImpl::RegisterProcess(FrameAvailableCallback callback, uint32_t sensor) {
    int32_t imageCaptureType = GetCaptureType(sensor);
    if (imageCaptureType < 0) {
        CAM_LOG_INFO << " Get null capture type. ";
    }
    threadDataStructs[sensor][imageCaptureType].on_data_frame = callback;
    return 0;
}

int32_t NvCameraImpl::GetImageData(std::string& image_data, uint32_t sensor) {
    SIPLStatus status = NVSIPL_STATUS_OK;
    int32_t imageCaptureType = GetCaptureType(sensor);
    if (imageCaptureType < 0) {
        CAM_LOG_INFO << " Get null capture type. ";
    }
    status = threadDataStructs[sensor][imageCaptureType].GetSIPLBuffer(image_data, sensor);
    if (NVSIPL_STATUS_OK == status) {
        return 0;
    } else {
        CAM_LOG_ERROR << "Get image data failed.";
        return -1;
    }
}

double NvCameraImpl::GetImageTimeStamp(uint32_t sensor) {
    int32_t imageCaptureType = GetCaptureType(sensor);
    if (imageCaptureType < 0) {
        CAM_LOG_INFO << " Get null capture type. ";
    }
    return threadDataStructs[sensor][imageCaptureType].image_info.time_stamp;
}

uint32_t NvCameraImpl::GetImageHeight(uint32_t sensor) {
    int32_t imageCaptureType = GetCaptureType(sensor);
    if (imageCaptureType < 0) {
        CAM_LOG_INFO << " Get null capture type. ";
    }
    return threadDataStructs[sensor][imageCaptureType].image_info.height;
}

uint32_t NvCameraImpl::GetImageWidth(uint32_t sensor) {
    int32_t imageCaptureType = GetCaptureType(sensor);
    if (imageCaptureType < 0) {
        CAM_LOG_INFO << " Get null capture type. ";
    }
    return threadDataStructs[sensor][imageCaptureType].image_info.width;
}

uint32_t NvCameraImpl::GetFrameID(uint32_t sensor) {
    int32_t imageCaptureType = GetCaptureType(sensor);
    if (imageCaptureType < 0) {
        CAM_LOG_INFO << " Get null capture type. ";
    }
    return threadDataStructs[sensor][imageCaptureType].image_info.frame_id;
}

}
}
}