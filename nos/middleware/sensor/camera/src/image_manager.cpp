#include "image_manager.hpp"

namespace hozon {
namespace netaos {
namespace camera {

// Number of images (buffers) to be allocated and registered for the capture output
static constexpr size_t CAPTURE_IMAGE_POOL_SIZE {6U};
// Number of images (buffers) to be allocated and registered for the ISP0 and ISP1 outputs
static constexpr size_t ISP_IMAGE_POOL_SIZE {4U};

SIPLStatus CImageManager::AllocateBuffers(ImagePool &imagePool)
{
    NvSciError err = NvSciError_Success;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> reconciledAttrList;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> conflictAttrList;

    reconciledAttrList.reset(new NvSciBufAttrList());
    conflictAttrList.reset(new NvSciBufAttrList());

    err = NvSciBufAttrListReconcile(imagePool.attrList.get(),
                                    1U,
                                    reconciledAttrList.get(),
                                    conflictAttrList.get());
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListReconcile");

    imagePool.sciBufObjs.resize(imagePool.size);
    for (size_t i = 0U; i < imagePool.size; i++) {
        err = NvSciBufObjAlloc(*reconciledAttrList, &(imagePool.sciBufObjs[i]));
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjAlloc");
        CHK_PTR_AND_RETURN(imagePool.sciBufObjs[i], "NvSciBufObjAlloc");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CImageManager::Allocate(uint32_t sensorId)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    for (uint32_t i = 0U; i < MAX_NUM_IMAGE_OUTPUTS; i++) {
        if (m_imagePools[sensorId][i].enable) {
            m_imagePools[sensorId][i].attrList.reset(new NvSciBufAttrList());
            NvSciError err = NvSciBufAttrListCreate(m_sciBufModule, m_imagePools[sensorId][i].attrList.get());
            CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListCreate");

            NvSciBufType bufType = NvSciBufType_Image;
            NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_Readonly;
            bool isCpuAcccessReq = true;
            bool isCpuCacheEnabled = true;
            NvSciBufAttrKeyValuePair attrKvp[] = {
                { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
                { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
                { NvSciBufGeneralAttrKey_NeedCpuAccess, &isCpuAcccessReq, sizeof(isCpuAcccessReq) },
                { NvSciBufGeneralAttrKey_EnableCpuCache, &isCpuCacheEnabled, sizeof(isCpuCacheEnabled) }
            };
            size_t uNumAttrs = (m_imagePools[sensorId][i].outputType
                == INvSIPLClient::ConsumerDesc::OutputType::ICP) ? 2U : 4U;
            err = NvSciBufAttrListSetAttrs(*m_imagePools[sensorId][i].attrList, attrKvp, uNumAttrs);
            CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");

            status = m_siplCamera->GetImageAttributes(sensorId,
                                                      m_imagePools[sensorId][i].outputType,
                                                      *m_imagePools[sensorId][i].attrList);

            CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::GetImageAttributes()");
            switch (m_imagePools[sensorId][i].outputType) {
                case INvSIPLClient::ConsumerDesc::OutputType::ICP:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP0:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP1:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP2:
                    status = AllocateBuffers(m_imagePools[sensorId][i]);
                    CHK_STATUS_AND_RETURN(status, "CImageManager::AllocateBuffers()");
                    break;
                default:
                    CAM_LOG_ERROR << "Unexpected output type";
                    return NVSIPL_STATUS_ERROR;
            }
        }
    }

    return status;
}

SIPLStatus CImageManager::Register(uint32_t sensorId)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    for (uint32_t i = 0U; i < MAX_NUM_IMAGE_OUTPUTS; i++) {
        if (m_imagePools[sensorId][i].enable) {
            switch (m_imagePools[sensorId][i].outputType) {
                case INvSIPLClient::ConsumerDesc::OutputType::ICP:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP0:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP1:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP2:
                    status = m_siplCamera->RegisterImages(sensorId,
                                                          m_imagePools[sensorId][i].outputType,
                                                          m_imagePools[sensorId][i].sciBufObjs);
                    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterImages()");
                    break;
                default:
                    CAM_LOG_ERROR << "Unexpected output type";
                    return NVSIPL_STATUS_ERROR;
            }
        }
    }

    return status;
}

SIPLStatus CImageManager::Init(INvSIPLCamera *siplCamera,
                               const std::unordered_map<uint32_t, NvSIPLPipelineConfiguration> &PieplineInfo)
{
    m_siplCamera = siplCamera;

    auto sciStatus = NvSciBufModuleOpen(&m_sciBufModule);
    CHK_NVSCISTATUS_AND_RETURN(sciStatus, "NvSciBufModuleOpen");

    sciStatus = NvSciSyncModuleOpen(&m_sciSyncModule);
    CHK_NVSCISTATUS_AND_RETURN(sciStatus, "NvSciSyncModuleOpen");

    for (auto &it : PieplineInfo) {
        m_imagePools[it.first][0].enable = true;
        m_imagePools[it.first][0].outputType = INvSIPLClient::ConsumerDesc::OutputType::ICP;
        m_imagePools[it.first][0].size = CAPTURE_IMAGE_POOL_SIZE;
        m_imagePools[it.first][1].enable = it.second.isp0OutputRequested;
        m_imagePools[it.first][1].outputType = INvSIPLClient::ConsumerDesc::OutputType::ISP0;
        m_imagePools[it.first][1].size = ISP_IMAGE_POOL_SIZE;
        m_imagePools[it.first][2].enable = it.second.isp1OutputRequested;
        m_imagePools[it.first][2].outputType = INvSIPLClient::ConsumerDesc::OutputType::ISP1;
        m_imagePools[it.first][2].size = ISP_IMAGE_POOL_SIZE;
        m_imagePools[it.first][3].enable = it.second.isp2OutputRequested;
        m_imagePools[it.first][3].outputType = INvSIPLClient::ConsumerDesc::OutputType::ISP2;
        m_imagePools[it.first][3].size = ISP_IMAGE_POOL_SIZE;
    }

    return NVSIPL_STATUS_OK;
}

void CImageManager::Deinit()
{
    if (m_sciSyncModule != nullptr) {
        NvSciSyncModuleClose(m_sciSyncModule);
    }
    if (m_sciBufModule != nullptr) {
        NvSciBufModuleClose(m_sciBufModule);
    }

    for (auto uSensorId = 0U; uSensorId < MAX_NUM_SENSORS; uSensorId++) {
        for (uint32_t i = 0U; i < MAX_NUM_IMAGE_OUTPUTS; i++) {
            if (m_imagePools[uSensorId][i].enable) {
                for (uint32_t j = 0U; j < m_imagePools[uSensorId][i].sciBufObjs.size(); j++) {
                    if (m_imagePools[uSensorId][i].sciBufObjs[j] == nullptr) {
                        CAM_LOG_WARN << "Attempt to free null NvSciBufObj.";
                        continue;
                    }
                    NvSciBufObjFree(m_imagePools[uSensorId][i].sciBufObjs[j]);
                }
                // Swap sciBufObjs vector with an equivalent empty vector to force deallocation
                std::vector<NvSciBufObj>().swap(m_imagePools[uSensorId][i].sciBufObjs);
            }
        }
    }
}

SIPLStatus CImageManager::GetBuffers(uint32_t uSensorId, INvSIPLClient::ConsumerDesc::OutputType outputType, std::vector<NvSciBufObj> &buffers)
{
    if (m_imagePools[uSensorId][(uint32_t)outputType].enable) {
        buffers = m_imagePools[uSensorId][(uint32_t)outputType].sciBufObjs;
        return NVSIPL_STATUS_OK;
    } else {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
}

void CImageManager::PrintISPOutputFormat(uint32_t pip,
                            INvSIPLClient::ConsumerDesc::OutputType outputType,
                            NvSciBufAttrList attrlist)
{
    NvSciBufAttrKeyValuePair keyVals[] = {
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },
    };

    size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
    if (NvSciBufAttrListGetAttrs(attrlist, keyVals, length) != NvSciError_Success) {
        CAM_LOG_ERROR << "NvSciBufAttrListGetAttrs Failed";
        return;
    }

    uint32_t planeCount = *(uint32_t*) keyVals[0].value;

    if (planeCount == 1) {
        NvSciBufAttrKeyValuePair keyVals[] = {
            { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneColorStd, NULL, 0 },
            { NvSciBufImageAttrKey_Layout, NULL, 0 },
        };

        size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
        if (NvSciBufAttrListGetAttrs(attrlist, keyVals, length) != NvSciError_Success) {
            CAM_LOG_ERROR << "NvSciBufAttrListGetAttrs Failed";
            return;
        }

        NvSciBufAttrValColorFmt planeColorFormat = *(NvSciBufAttrValColorFmt*)keyVals[0].value;
        NvSciBufAttrValColorStd planeColorStd = *(NvSciBufAttrValColorStd*)keyVals[1].value;
        NvSciBufAttrValImageLayoutType layoutType =
                *(NvSciBufAttrValImageLayoutType*)keyVals[2].value;

        std::string layout = "PL";
        std::string colorStd = "SENSOR_RGBA";
        std::string colorFormat = "RGBA PACKED FLOAT16";
        if (layoutType == NvSciBufImage_BlockLinearType) {
            layout = "BL";
        }
        if (planeColorStd == NvSciColorStd_REC709_ER) {
            colorStd = "REC709_ER";
        } else if (planeColorStd == NvSciColorStd_SRGB) {
            colorStd = "SRGB";
        }
        if (planeColorFormat == NvSciColor_Y16) {
            colorFormat = "LUMA PACKED UINT16";
        } else if (planeColorFormat == NvSciColor_A8Y8U8V8) {
            colorFormat = "VUYX PACKED UINT8";
        } else if (planeColorFormat == NvSciColor_A16Y16U16V16) {
            colorFormat = "VUYX PACKED UINT16";
        }

        CAM_LOG_INFO << "Pipeline: " << pip
                << " ISP Output: " << ((uint32_t)outputType - 1)
                << " is using "
                << colorFormat << " " << layout << " " << colorStd << "\n";

    } else {
        NvSciBufAttrKeyValuePair keyVals[] = {
            { NvSciBufImageAttrKey_SurfMemLayout, NULL, 0 },
            { NvSciBufImageAttrKey_SurfColorStd, NULL, 0 },
            { NvSciBufImageAttrKey_Layout, NULL, 0 },
            { NvSciBufImageAttrKey_SurfSampleType, NULL, 0 },
            { NvSciBufImageAttrKey_SurfBPC, NULL, 0 },
        };

        size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
        if (NvSciBufAttrListGetAttrs(attrlist, keyVals, length) != NvSciError_Success) {
            CAM_LOG_ERROR << "NvSciBufAttrListGetAttrs Failed";
            return;
        }

        NvSciBufSurfMemLayout surfMemLayout = *(NvSciBufSurfMemLayout*)keyVals[0].value;
        if (surfMemLayout != NvSciSurfMemLayout_SemiPlanar) {
            CAM_LOG_ERROR << "Only Semi Planar surfaces are supported with Surf attributes";
            return;
        }

        NvSciBufAttrValColorStd surfColorStd = *(NvSciBufAttrValColorStd*)keyVals[1].value;
        if (surfColorStd != NvSciColorStd_REC709_ER) {
            CAM_LOG_ERROR << "Color standard not supported for YUV images";
            return;
        }
        NvSciBufAttrValImageLayoutType layoutType =
                *(NvSciBufAttrValImageLayoutType*)keyVals[2].value;
        NvSciBufSurfSampleType surfSampleType = *(NvSciBufSurfSampleType*)keyVals[3].value;
        NvSciBufSurfBPC surfBPC = *(NvSciBufSurfBPC*)keyVals[4].value;

        std::string layout = "PL";
        std::string sampleType = "420";
        std::string bpc = "8";

        if (layoutType == NvSciBufImage_BlockLinearType) {
            layout = "BL";
        }
        if (surfSampleType == NvSciSurfSampleType_444) {
            sampleType = "444";
        }
        if (surfBPC == NvSciSurfBPC_16) {
            bpc = "16";
        }
        CAM_LOG_INFO << "-------------Pipeline: " << pip
                << " ISP Output: " << ((uint32_t)outputType - 1)
                << " is using "
                << "YUV " << sampleType << " SEMI-PLANAR UINT" << bpc
                << " " << layout << " REC_709ER" << "\n";
    }

}

SIPLStatus CImageManager::SetNvSciSyncCPUWaiter(uint32_t pip, bool isp0Enabled, bool isp1Enabled, bool isp2Enabled)
{
    if (!isp0Enabled && !isp1Enabled && !isp2Enabled)
    {
        return NVSIPL_STATUS_OK;
    }

    NvSciSyncAttrList signalerAttrList;
    NvSciSyncAttrList waiterAttrList;

    // SIPL signalers across all ISP outputs will be the same and thus only need to create a
    // single attribute list
    auto sciErr = NvSciSyncAttrListCreate(m_sciSyncModule,
                                            &signalerAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer signaler NvSciSyncAttrListCreate");

    // All ISP outputs have the same signaler attributes, so just using ISP0
    auto status = m_siplCamera->FillNvSciSyncAttrList(pip,
                                                    INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                    signalerAttrList,
                                                    SIPL_SIGNALER);
    CHK_STATUS_AND_RETURN(status, "Producer signaler INvSIPLCamera::FillNvSciSyncAttrList");

    // Waiters across all ISP outputs will be the same for our purposes (CPU wait) and thus
    // only need a single attribute list to represent all consumers
    sciErr = NvSciSyncAttrListCreate(m_sciSyncModule,
                                        &waiterAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer waiter NvSciSyncAttrListCreate");

    // Create application's NvSciSync attributes for CPU waiting, reconcile with SIPL's signaler
    // attributes, create NvSciSyncObj with the reconciled attributes, register the object with
    // SIPL as EOF sync obj
    NvSciSyncAttrList unreconciledLists[2];
    NvSciSyncAttrList reconciledList = NULL;
    NvSciSyncAttrList conflictList = NULL;

    NvSciSyncAttrKeyValuePair keyValue[3];
    bool cpuWaiter = true;
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void *)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    bool requireTimestamp = true;
    keyValue[2].attrKey = NvSciSyncAttrKey_WaiterRequireTimestamps;
    keyValue[2].value = (void*)&requireTimestamp;
    keyValue[2].len = sizeof(requireTimestamp);
    sciErr = NvSciSyncAttrListSetAttrs(waiterAttrList, keyValue, 3);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListSetAttrs");

    size_t inputCount = 0U;
    unreconciledLists[inputCount++] = signalerAttrList;
    unreconciledLists[inputCount++] = waiterAttrList;

    // Reconcile the  waiter and signaler through the unreconciledLists
    sciErr = NvSciSyncAttrListReconcile(unreconciledLists,
                                        inputCount,
                                        &reconciledList,
                                        &conflictList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler and waiter NvSciSyncAttrListReconcile");

    NvSciSyncAttrListFree(signalerAttrList);
    NvSciSyncAttrListFree(waiterAttrList);

    NvSciSyncObj syncObj;

    // Allocate the sync object
    sciErr = NvSciSyncObjAlloc(reconciledList, &syncObj);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Reconciled signaler and waiter NvSciSyncObjAlloc");

    NvSciSyncAttrListFree(reconciledList);
    if (conflictList != nullptr) {
        NvSciSyncAttrListFree(conflictList);
    }

    // Register with SIPL, SIPL expects to register only one NvSciSyncObj for all the enabled
    // ISP outputs of a given pipeline
    status = m_siplCamera->RegisterNvSciSyncObj(pip,
                                                INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                NVSIPL_EOFSYNCOBJ,
                                                syncObj);
    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");

    NvSciSyncObjFree(syncObj);

    return NVSIPL_STATUS_OK;
}

CImageManager::~CImageManager()
{
    Deinit();
}

}
}
}