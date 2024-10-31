/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include "NvSIPLCamera.hpp"
#include "NvSIPLPipelineMgr.hpp"
#include "CUtils.hpp"
#include "CChannel.hpp"
#include "CIpcProducerChannel.hpp"

#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "CSensorRegInf.hpp"


#ifndef CMASTER_HPP
#define CMASTER_HPP

using namespace std;
using namespace nvsipl;

/** CMaster class */
class CMaster
{
 public:
    CMaster(AppType appType):
        m_appType(appType)
    {
    }

    ~CMaster(void)
    {
        //need to release other nvsci resources before closing modules.
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i].reset();
            }
        }

        LOG_DBG("CMaster release.\n");

        if (m_sciBufModule != nullptr) {
          NvSciBufModuleClose(m_sciBufModule);
        }

        if (m_sciSyncModule != nullptr) {
          NvSciSyncModuleClose(m_sciSyncModule);
        }

        NvSciIpcDeinit();
    }

    SIPLStatus Setup(uint32_t multiNum)
    {
        // Camera Master setup
        m_upCamera = INvSIPLCamera::GetInstance();
        CHK_PTR_AND_RETURN(m_upCamera, "INvSIPLCamera::GetInstance()");

        auto sciErr = NvSciBufModuleOpen(&m_sciBufModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufModuleOpen");

        sciErr = NvSciSyncModuleOpen(&m_sciSyncModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

    
        sciErr = NvSciIpcInit();
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcInit");

        multicastNum = multiNum;
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetPlatformConfig(PlatformCfg* pPlatformCfg, NvSIPLDeviceBlockQueues &queues)
    {
        m_platformCfg = *pPlatformCfg;
        return m_upCamera->SetPlatformCfg(pPlatformCfg, queues);
    }

    SIPLStatus SetPipelineConfig(uint32_t uIndex, NvSIPLPipelineConfiguration &pipelineCfg, NvSIPLPipelineQueues &pipelineQueues)
    {
        return m_upCamera->SetPipelineCfg(uIndex, pipelineCfg, pipelineQueues);
    }

    SIPLStatus InitPipeline()
    {
        auto status = m_upCamera->Init();
        CHK_STATUS_AND_RETURN(status, "m_upCamera->Init()");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus StartStream(void)
    {
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i]->Start();
            }
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus StartPipeline(void)
    {
        const SIPLStatus status = m_upCamera->Start();
        CHK_STATUS_AND_RETURN(status, "Start SIPL");
        return NVSIPL_STATUS_OK;
    }

    void StopStream(const uint32_t index)
    {
            if (nullptr != m_upChannels[index]) {
                m_upChannels[index]->Stop();
            }

            if (nullptr != m_upChannels[index]) {
                m_upChannels[index].reset();
            }
    }

    SIPLStatus AsyncStopStream(void)
    {
        auto ret = NVSIPL_STATUS_OK;
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]){
                m_upStopChannelThreads[i].reset(new std::thread(&CMaster::StopStream, this,i));
                if(m_upStopChannelThreads[i] == nullptr){
                    LOG_ERR("Failed to create StopStream thread\n");
                    ret = NVSIPL_STATUS_ERROR;
                    break;
                }
            }
        }
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if(m_upStopChannelThreads[i] != nullptr){
                m_upStopChannelThreads[i]->join();
                m_upStopChannelThreads[i].reset();
            }
        }
        return ret;
    }

    SIPLStatus StopPipeline(void)
    {
        const SIPLStatus status = m_upCamera->Stop();
        CHK_STATUS_AND_RETURN(status, "Stop SIPL");

        return NVSIPL_STATUS_OK;
    }

    void DeinitPipeline(void)
    {
        auto status = m_upCamera->Deinit();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLCamera::Deinit failed. status: %x\n", status);
        }
    }

    SIPLStatus RegisterSource(SensorInfo *pSensorInfo, CProfiler *pProfiler,uint32_t index)
    {
        LOG_DBG("CMaster: RegisterSource.\n");

        if (nullptr == pSensorInfo || nullptr == pProfiler) {
            LOG_ERR("%s: nullptr\n", __func__);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        if (pSensorInfo->id >= MAX_NUM_SENSORS) {
            LOG_ERR("%s: Invalid sensor id: %u\n", __func__, pSensorInfo->id);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        m_upChannels[pSensorInfo->id] = CreateChannel(pSensorInfo, pProfiler,index*multicastNum,multicastNum);
        CHK_PTR_AND_RETURN(m_upChannels[pSensorInfo->id], "Master CreateChannel");
        m_upChannels[pSensorInfo->id]->Init();

        auto status = m_upChannels[pSensorInfo->id]->CreateBlocks(pProfiler);
        CHK_STATUS_AND_RETURN(status, "Master CreateBlocks");

        return NVSIPL_STATUS_OK;
    }

    void CreateStream(const uint32_t index){
        string tname = "CreateStream"+std::to_string(index);
        pthread_setname_np(pthread_self(), tname.c_str());
        auto status = m_upChannels[index]->Connect();
        isInitStreamSuces[index] = false;
        if(status == NVSIPL_STATUS_OK){
            status = m_upChannels[index]->InitBlocks();
            if(status == NVSIPL_STATUS_OK){
                status = m_upChannels[index]->Reconcile();
                if(status == NVSIPL_STATUS_OK)
                    isInitStreamSuces[index] = true;
            }         
        }
        return;
    }

    SIPLStatus AsyncInitStreams(void)
    {
        LOG_DBG("CMaster: InitStream.\n");
        
        auto ret = NVSIPL_STATUS_OK;
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]){
                m_upInitChannelThreads[i].reset(new std::thread(&CMaster::CreateStream, this,i));
                if(m_upInitChannelThreads[i] == nullptr){
                    LOG_ERR("Failed to create InitStream thread\n");
                    ret = NVSIPL_STATUS_ERROR;
                    break;
                }
            }
        }
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if(m_upInitChannelThreads[i] != nullptr){
                m_upInitChannelThreads[i]->join();
                m_upInitChannelThreads[i].reset();
                if(!isInitStreamSuces[i]){
                    LOG_ERR("Init sersorID:%d stream channel fail!",i);
                    ret = NVSIPL_STATUS_ERROR;
                }
            }
        }
        return ret;
    }
    

    SIPLStatus InitStream(void)
    {
        LOG_DBG("CMaster: InitStream.\n");

        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                auto status = m_upChannels[i]->Connect();
                CHK_STATUS_AND_RETURN(status, "CMaster: Channel connect.");

                status = m_upChannels[i]->InitBlocks();
                CHK_STATUS_AND_RETURN(status, "InitBlocks");

                status = m_upChannels[i]->Reconcile();
                CHK_STATUS_AND_RETURN(status, "Channel Reconcile");
            }
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus OnFrameAvailable(uint32_t uSensor, NvSIPLBuffers &siplBuffers)
    {
        if (uSensor >= MAX_NUM_SENSORS) {
            LOG_ERR("%s: Invalid sensor id: %u\n", __func__, uSensor);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        if (m_appType == IPC_SIPL_PRODUCER) {
            if(uSensor < MAX_NUM_SENSORS_7V){
                INvSIPLClient::INvSIPLNvMBuffer* pNvMBuf = reinterpret_cast<INvSIPLClient::INvSIPLNvMBuffer*>(siplBuffers[0].second);
                const INvSIPLClient::ImageMetaData &md = pNvMBuf->GetImageData();
                if(!m_dropinit){ // get first frame tsc of 7v 
                    std::unique_lock<std::mutex> lock(m_mutex);
                    if(!m_dropinit){
                        m_startframetsc = (md.frameCaptureTSC << 5) / 1000 - 33333 >> 1; //get ns
                        m_dropinit = 1;
                    }
                }
                m_frameNum[uSensor] = (((md.frameCaptureTSC << 5) / 1000) - m_startframetsc) / 33333;
                if((m_frameNum[uSensor] % 3) != 0){
                    return NVSIPL_STATUS_OK;
                }
            }
            CIpcProducerChannel* pIpcProducerChannel = dynamic_cast<CIpcProducerChannel*>(m_upChannels[uSensor].get());
            return pIpcProducerChannel->Post(siplBuffers);
        } else {
            LOG_WARN("Received unexpected OnFrameAvailable, appType: %u\n", m_appType);
            return NVSIPL_STATUS_ERROR;
        }
    }

    SIPLStatus GetMaxErrorSize(const uint32_t devBlkIndex, size_t &size)
    {
        return m_upCamera->GetMaxErrorSize(devBlkIndex, size);
    }

    SIPLStatus GetErrorGPIOEventInfo(const uint32_t devBlkIndex,
                                     const uint32_t gpioIndex,
                                     SIPLGpioEvent &event)
    {
        return m_upCamera->GetErrorGPIOEventInfo(devBlkIndex, gpioIndex, event);
    }

    SIPLStatus GetDeserializerErrorInfo(const uint32_t devBlkIndex,
                                        SIPLErrorDetails * const deserializerErrorInfo,
                                        bool & isRemoteError,
                                        uint8_t& linkErrorMask)
    {
        return m_upCamera->GetDeserializerErrorInfo(devBlkIndex, deserializerErrorInfo,
                                                   isRemoteError, linkErrorMask);
    }

    SIPLStatus GetModuleErrorInfo(const uint32_t index,
                                         SIPLErrorDetails * const serializerErrorInfo,
                                         SIPLErrorDetails * const sensorErrorInfo)
    {
        return m_upCamera->GetModuleErrorInfo(index, serializerErrorInfo, sensorErrorInfo);
    }

    SIPLStatus RegisterAutoControl(uint32_t uIndex, PluginType type, ISiplControlAuto* customPlugin, std::vector<uint8_t>& blob)
    {
        return m_upCamera->RegisterAutoControlPlugin(uIndex, type, customPlugin, blob);
    }

    SIPLStatus DisableLink(uint32_t index)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_upCamera->DisableLink(index);
    }

    SIPLStatus EnableLink(uint32_t index, bool resetModule)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_upCamera->EnableLink(index, resetModule);
    }

    SIPLStatus ReadEEPROMData(const uint32_t index,const uint16_t address,const uint32_t length,uint8_t * const buffer){
        LOG_DBG("--ReadEEPROMData start\n");
        auto status = m_upCamera->ReadEEPROMData(index, address, length, buffer);
        LOG_DBG("--ReadEEPROMData end\n");
        return status;
    }

    Sensor_CustomInterface* CustomInterfaceList[16] = {nullptr};
    Sensor_CustomInterface* GetSensor_CustomInterface(const uint32_t uSensor) {
        IInterfaceProvider* moduleInterfaceProvider = nullptr;
        Sensor_CustomInterface* SensorCustomInterface = nullptr;

        // Get the interface provider
        SIPLStatus status = m_upCamera->GetModuleInterfaceProvider(uSensor,
            moduleInterfaceProvider);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Error %d while getting module interface provider for sensor ID: %d\n",
                status, uSensor);
        } else if (moduleInterfaceProvider != nullptr) {
            // Get the custom interface and cast
            SensorCustomInterface = static_cast<Sensor_CustomInterface*>
                (moduleInterfaceProvider->GetInterface(SENSOR_CUSTOM_INTERFACE_ID));
            if (SensorCustomInterface != nullptr) {
                // Verify that the ID matches expected - we have the correct custom interface
                if (SensorCustomInterface->getInstanceInterfaceID() ==
                    SENSOR_CUSTOM_INTERFACE_ID) {
                    LOG_DBG("sensor custom interface found\n");
                } else {
                    LOG_ERR("Incorrect interface obtained from module\n");
                    // Set the return pointer to NULL because the obtained pointer
                    // does not point to correct interface.
                    SensorCustomInterface = nullptr;
                }
            }
        }

        return SensorCustomInterface;
    }


    SIPLStatus CheckSensorStatus(const uint32_t uSensor)
    {
        const uint32_t index_ns = uSensor;
        uint32_t moduleIndex = 0U;
        uint32_t devBlkIndex = 0U;
        bool bFound = false;
        // Get devBlkIndex and linkIndex
        for (uint32_t d = 0U; d != m_platformCfg.numDeviceBlocks; d++) {
            const DeviceBlockInfo& devblock = m_platformCfg.deviceBlockList[d];
            for (uint32_t m = 0U; m != devblock.numCameraModules; m++) {
                const CameraModuleInfo &mod = devblock.cameraModuleInfoList[m];
                const SensorInfo &sensor = mod.sensorInfo;
                if (sensor.id == index_ns) {
                    moduleIndex = m;
                    devBlkIndex = d;
                    bFound = true;
                }
            }
        }
        if (!bFound)
            return NVSIPL_STATUS_NOT_SUPPORTED;
        if (m_customInterface[uSensor] == NULL)
            return NVSIPL_STATUS_NOT_SUPPORTED;

        if (m_platformCfg.deviceBlockList[devBlkIndex].cameraModuleInfoList[moduleIndex].sensorInfo.name.compare("AR0820") == 0) {
            // AR0820NonFuSaCustomInterface* ar0820Interface = static_cast<AR0820NonFuSaCustomInterface*>(m_customInterface[uSensor]);
            // return ar0820Interface->CheckModuleStatus();
        } else if (m_platformCfg.deviceBlockList[devBlkIndex].cameraModuleInfoList[moduleIndex].sensorInfo.name.compare("OV2311") == 0) {
            // OV2311NonFuSaCustomInterface* ov2311Interface = static_cast<OV2311NonFuSaCustomInterface*>(m_customInterface[uSensor]);
            // return ov2311Interface->CheckModuleStatus();
        }
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus InitCustomInterfaces()
    {
        IInterfaceProvider* moduleInterfaceProvider = nullptr;
        Interface *customInterface = nullptr;
        for (auto d = 0u; d != m_platformCfg.numDeviceBlocks; d++) {
            auto db = m_platformCfg.deviceBlockList[d];
            for (auto m = 0u; m != db.numCameraModules; m++) {
                auto module = db.cameraModuleInfoList[m];
                auto sensor = module.sensorInfo;
                uint32_t uSensor = sensor.id;
                SIPLStatus status = m_upCamera->GetModuleInterfaceProvider(uSensor, moduleInterfaceProvider);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("Error %d while getting module interface provider for sensor ID: %d\n",
                        status, uSensor);
                } else if (moduleInterfaceProvider != nullptr) {
                    UUID uuid;
                    // if(m_platformCfg.deviceBlockList[d].cameraModuleInfoList[m].sensorInfo.name.compare("AR0820") == 0)
                    //     uuid = AR0820_NONFUSA_CUSTOM_INTERFACE_ID;
                    // else if(m_platformCfg.deviceBlockList[d].cameraModuleInfoList[m].sensorInfo.name.compare("OV2311") == 0)
                    //     uuid = OV2311_NONFUSA_CUSTOM_INTERFACE_ID;
                    customInterface = moduleInterfaceProvider->GetInterface(uuid);
                    if (customInterface != nullptr) {
                        // Verify that the ID matches expected - we have the correct custom interface
                        if (customInterface->getInstanceInterfaceID() == uuid) {
                            LOG_DBG("Custom interface found\n");
                            m_customInterface[uSensor] = customInterface;
                        } else {
                            LOG_ERR("Incorrect interface obtained from module\n");
                            // Set the return pointer to NULL because the obtained pointer
                            // does not point to correct interface.
                            customInterface = nullptr;
                            m_customInterface[uSensor] = NULL;
                        }
                    }
                }
            }
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus GetSensorRegData(CSensorRegInf* upsensorInf, const std::string uSensorName, const uint32_t uSensor) {
        upsensorInf->InterfaceInitRegister(uSensor, &upsensorInf->interfaceregister, m_upCamera.get());
        return upsensorInf->GetSensorRegisterInfo(uSensorName, uSensor);
    }

    void DeInitCustomInterfaces(CSensorRegInf* upsensorInf) {
        upsensorInf->Deinit();
    }

    SIPLStatus AttachConsumer(uint32_t uSensor, uint32_t index)
    {
        if (nullptr != m_upChannels[uSensor]) {
            CIpcProducerChannel *pIpcProducerChannel = dynamic_cast<CIpcProducerChannel *>(m_upChannels[uSensor].get());
            pIpcProducerChannel->attach(index);
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus DetachConsumer(uint32_t uSensor, uint32_t index)
    {
        if (nullptr != m_upChannels[uSensor]) {
            CIpcProducerChannel *pIpcProducerChannel = dynamic_cast<CIpcProducerChannel *>(m_upChannels[uSensor].get());
            pIpcProducerChannel->detach(index);
        }
        return NVSIPL_STATUS_OK;
    }

private:
    std::unique_ptr<CChannel> CreateChannel(SensorInfo *pSensorInfo, CProfiler *pProfiler,uint32_t channelStrIndex,uint32_t multicastNum)
    {
            return std::unique_ptr<CIpcProducerChannel>(
                    new CIpcProducerChannel(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_upCamera.get(),channelStrIndex,multicastNum));
    }
    std::mutex m_mutex;
    AppType m_appType;
    unique_ptr<INvSIPLCamera> m_upCamera {nullptr};
    NvSciSyncModule m_sciSyncModule {nullptr};
    NvSciBufModule m_sciBufModule {nullptr};
    unique_ptr<CChannel> m_upChannels[MAX_NUM_SENSORS] {nullptr};
    bool isInitStreamSuces[MAX_NUM_SENSORS] {false};
    unique_ptr<thread> m_upInitChannelThreads[MAX_NUM_SENSORS] {nullptr};
    unique_ptr<thread> m_upStopChannelThreads[MAX_NUM_SENSORS] {nullptr};
    uint32_t multicastNum = 1;
    PlatformCfg m_platformCfg;

    Interface *m_customInterface[MAX_NUM_SENSORS] = {NULL};

    uint64_t m_frameNum[MAX_NUM_SENSORS] = {0U};
    uint8_t m_dropinit = 0;
    uint64_t m_startframetsc = 0;
};

#endif //CMASTER_HPP
