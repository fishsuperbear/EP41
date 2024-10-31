#include "CVirtualCamProducer.hpp"

#include <algorithm>
#include <fstream>
#include "CPoolManager.hpp"
#include "scibuf_utils.h"
#include "video_utils.h"

/* NvMediaIDE only supports input surface formats which have 2 planes */
#define IDE_APP_MAX_INPUT_PLANE_COUNT 2U
#define IDE_APP_BASE_ADDR_ALIGN 256U

constexpr static int32_t OUTPUT_TYPE_UNDEFINED = -1;

CVirtualCamProducer::CVirtualCamProducer(NvSciStreamBlock handle, PicInfo* pic_info) : CProducer("CVirtualCamProducer", handle, pic_info->sid), pic_info_(pic_info) {
    // m_vIspBufObjs.resize(MAX_NUM_PACKETS);

    buffer_pool_.reset(new BufferPool(MAX_NUM_PACKETS, m_uSensorId));

    // std::string file_path = "cam" + std::to_string(m_uSensorId) + ".265";
    // file_.reset(new std::ofstream(file_path, std::ios::binary | std::ios::out));
}

CVirtualCamProducer::~CVirtualCamProducer(void) {
    PLOG_DBG("Release.\n");

    // for (NvSciBufObj bufObj : m_vIspBufObjs) {
    //     if (bufObj != nullptr) {
    //         NvSciBufObjFree(bufObj);
    //     }
    // }
    // std::vector<NvSciBufObj>().swap(m_vIspBufObjs);
}

void CVirtualCamProducer::PreInit(std::shared_ptr<CLateConsumerHelper> lateConsHelper) {
    m_spLateConsHelper = lateConsHelper;
}

SIPLStatus CVirtualCamProducer::HandleClientInit(void) {

    PLOG_DBG("NvMediaIDECreate--codec=%d,width=%d,height=%d\n", pic_info_->codec, pic_info_->width, pic_info_->height);

    decoder_ = NvMediaIDECreate((NvMediaVideoCodec)pic_info_->codec,  // codec
                                pic_info_->width,                     // width
                                pic_info_->height,                    // height
                                3,                                    // maxReferences
                                6291456,                              // maxBitstreamSize
                                1,                                    // inputBuffering
                                0,                                    // decoder flags
                                NVMEDIA_DECODER_INSTANCE_0);          // instance ID

    static NvMediaParserClientCb client_cb{0};
    client_cb.BeginSequence = &CVirtualCamProducer::ParserCb_BeginSequence;
    client_cb.DecodePicture = &CVirtualCamProducer::ParserCb_DecodePicture;
    client_cb.DisplayPicture = &CVirtualCamProducer::ParserCb_DisplayPicture;
    client_cb.UnhandledNALU = &CVirtualCamProducer::ParserCb_UnhandledNALU;
    client_cb.AllocPictureBuffer = &CVirtualCamProducer::ParserCb_AllocPictureBuffer;
    client_cb.Release = &CVirtualCamProducer::ParserCb_Release;
    client_cb.AddRef = &CVirtualCamProducer::ParserCb_AddRef;
    client_cb.CreateDecrypter = nullptr;
    client_cb.DecryptHdr = nullptr;
    client_cb.SliceDecode = nullptr;
    client_cb.GetClearHdr = nullptr;
    client_cb.GetBackwardUpdates = &CVirtualCamProducer::ParserCb_GetBackwardUpdates;
    client_cb.GetDpbInfoForMetadata = nullptr;
    static NvMediaParserParams parser_params{0};
    parser_params.pClient = &client_cb;
    parser_params.pClientCtx = this;
    parser_params.uErrorThreshold = 50;
    parser_params.uReferenceClockRate = 0;
    parser_params.eCodec = NVMEDIA_VIDEO_CODEC_HEVC;
    media_parser_ = NvMediaParserCreate(&parser_params);
    // bool enable_vc1ap_interlaced = false;
    // float default_frame_rate = 30;
    // NvMediaParserSetAttribute(media_parser_, NvMParseAttr_EnableVC1APInterlaced, sizeof(bool), &enable_vc1ap_interlaced);
    // NvMediaParserSetAttribute(media_parser_, NvMParseAttr_SetDefaultFramerate, sizeof(float), &default_frame_rate);
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::HandleStreamInit(void) {
    SIPLStatus status = CProducer::HandleStreamInit();
    // If enable lateAttach, the comsumer count queried is the value passed into multicast block when created,
    // including late consumer, but the producer only need to handle the early consumer's sync obj,
    // hence the m_numWaitSyncObj should reduce by late consumer count.
    if (m_spLateConsHelper && m_spLateConsHelper->GetLateConsCount() > 0) {
        m_numWaitSyncObj -= m_spLateConsHelper->GetLateConsCount();
    }
    return status;
}

// Create and set CPU signaler and waiter attribute lists.
SIPLStatus CVirtualCamProducer::SetSyncAttrList(PacketElementType userType, NvSciSyncAttrList& signalerAttrList, NvSciSyncAttrList& waiterAttrList) {
    // INvSIPLClient::ConsumerDesc::OutputType outputType;
    // auto status = MapElemTypeToOutputType(userType, outputType);
    // PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    auto status = NvMediaIDEFillNvSciSyncAttrList(decoder_, signalerAttrList, NVMEDIA_SIGNALER);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to fill signaler attr list.[%d]\n", status);
        return NVSIPL_STATUS_ERROR;
    }

    NvSciSyncAttrKeyValuePair keyValue[2];
    bool cpuWaiter = true;
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    auto sciErr = NvSciSyncAttrListSetAttrs(waiterAttrList, keyValue, 2);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

// SIPLStatus CVirtualCamProducer::CreateBufAttrLists(NvSciBufModule bufModule) {
//     LOG_DBG("CreateBufAttrLists\n");
//     INvSIPLClient::ConsumerDesc::OutputType outputType;
//     auto status = MapElemTypeToOutputType(userType, outputType);
//     PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");
//     // Create ISP buf attrlist
//     // auto status = CClientCommon::CreateBufAttrLists(bufModule);
//     // CHK_STATUS_AND_RETURN(status, "CClientCommon::CreateBufAttrList");

//     // NvSciBufAttrListClone(m_bufAttrLists[0], &(vd_->GetCtx()->bufAttributeList));
//     return NVSIPL_STATUS_OK;
// }

SIPLStatus CVirtualCamProducer::SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList& bufAttrList) {
    // INvSIPLClient::ConsumerDesc::OutputType outputType;
    // auto status = MapElemTypeToOutputType(userType, outputType);
    // PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    auto status = SetBufAttrList(bufAttrList);
    PCHK_STATUS_AND_RETURN(status, "SetBufAttrList for ISP");

    return NVSIPL_STATUS_OK;
}

// Buffer setup functions
SIPLStatus CVirtualCamProducer::SetBufAttrList(NvSciBufAttrList& bufAttrList) {
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair attrKvp = {NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm)};
    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, &attrKvp, 1);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    bool isCpuAcccessReq = true;
    bool isCpuCacheEnabled = true;

    NvSciBufAttrKeyValuePair setAttrs[] = {
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &isCpuAcccessReq, sizeof(isCpuAcccessReq)},
        {NvSciBufGeneralAttrKey_EnableCpuCache, &isCpuCacheEnabled, sizeof(isCpuCacheEnabled)},
    };
    sciErr = NvSciBufAttrListSetAttrs(bufAttrList, setAttrs, 2);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    // fill ide buf attrlist
    GetIdeImageAttributes(bufAttrList);

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CVirtualCamProducer::MapDataBuffer(PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj) {
    PLOG_DBG("Mapping data buffer, packetIndex: %u.\n", packetIndex);
    if (!buffer_pool_->MapBuffer(packetIndex, bufObj)) {
        PLOG_ERR("MapBuffer error!");
    }

    auto status = NvMediaIDERegisterNvSciBufObj(decoder_, bufObj);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main:failed to register NvSciBufObj\n");
    }

    // auto sciErr = NvSciBufObjDup(m_packets[packetIndex].dataObj, &m_vIspBufObjs[packetIndex]);
    // LOG_DBG("MapDataBuffer    %p/%p\n", m_packets[packetIndex].dataObj, m_vIspBufObjs[packetIndex]);
    // PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjDup");

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CVirtualCamProducer::MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj) {
    PLOG_DBG("Mapping meta buffer, packetIndex: %u.\n", packetIndex);
    auto sciErr = NvSciBufObjGetCpuPtr(bufObj, (void**)&m_metaPtrs[packetIndex]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetCpuPtr");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj) {
    // vd_->RegisterSignalSyncObj(m_signalSyncObj);
    // Only one signalSyncObj
    // auto status = m_pCamera->RegisterNvSciSyncObj(m_uSensorId, m_ispOutputType, NVSIPL_EOFSYNCOBJ, m_signalSyncObj);
    // PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");

    NvMediaStatus nvMediaStatus = NvMediaIDERegisterNvSciSyncObj(decoder_, NVMEDIA_EOFSYNCOBJ, signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvMediaStatus, "NvMediaIDERegisterNvSciSyncObj failed.");

    nvMediaStatus = NvMediaIDESetNvSciSyncObjforEOF(decoder_, signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvMediaStatus, "NvMediaIDESetNvSciSyncObjforEOF failed.");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj) {
#ifdef NVMEDIA_QNX
    auto status = m_pCamera->RegisterNvSciSyncObj(m_uSensorId, m_ispOutputType, NVSIPL_PRESYNCOBJ, m_waiterSyncObjs[index]);
    PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");
#endif  // NVMEDIA_QNX \
    // In streaming phase, can not register sync obj into camera
    if (m_streamPhase != StreamPhase_Initialization) {
        PLOG_DBG("Camera only support registering sync obj in initialization phase. \n");
        return NVSIPL_STATUS_OK;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::HandleSetupComplete(void) {
    auto status = CProducer::HandleSetupComplete();
    PCHK_STATUS_AND_RETURN(status, "HandleSetupComplete");

    status = RegisterBuffers();
    PCHK_STATUS_AND_RETURN(status, "RegisterBuffers");

    is_setup_complete_ = true;

    return NVSIPL_STATUS_OK;
}

bool CVirtualCamProducer::IsComplete() {
    return m_streamPhase == StreamPhase_Streaming;
}

SIPLStatus CVirtualCamProducer::RegisterBuffers(void) {
    // PLOG_DBG("RegisterBuffers\n");
    // for (auto& obj : m_vIspBufObjs) {
    //     auto status = NvMediaIDERegisterNvSciBufObj(decoder_, obj);
    //     if (status != NVMEDIA_STATUS_OK) {
    //         LOG_DBG("main:failed to register NvSciBufObj\n");
    //     }
    // }
    return NVSIPL_STATUS_OK;
}

// Before calling PreSync, m_nvmBuffers[packetIndex] should already be filled.
SIPLStatus CVirtualCamProducer::InsertPrefence(PacketElementType userType, uint32_t packetIndex, NvSciSyncFence& prefence) {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::GetPostfence(uint32_t packetIndex, NvSciSyncFence* pPostfence) {
    // auto status = m_nvmBuffers[packetIndex]->GetEOFNvSciSyncFence(pPostfence);
    // PCHK_STATUS_AND_RETURN(status, "GetEOFNvSciSyncFence");

    return NVSIPL_STATUS_OK;
}

void CVirtualCamProducer::OnPacketGotten(uint32_t packetIndex) {
    PLOG_INFO("OnPacketGotten  ReleaseBuffer  idx=%d\n", packetIndex);
    if (!buffer_pool_->ReleaseBuffer(packetIndex)) {
        LOG_ERR("ReleaseBuffer false\n");
    }
    // m_nvmBuffers[packetIndex]->Release();
}

static uint64_t TscToRealtimeUs(uint64_t tsc) {
    timespec mono_time = {0};
    clock_gettime(CLOCK_MONOTONIC, &mono_time);
    // if (ret != TIME_OK) {
    //     NVS_LOG_ERROR << "Fail to get mono time, err " << errno << ", " << ret;
    // }

    timespec real_time = {0};
    clock_gettime(CLOCK_REALTIME, &real_time);
    // if (ret != TIME_OK) {
    //     NVS_LOG_ERROR << "Fail to get real time, err " << errno << ", " << ret;
    // }

    int64_t diff_us = real_time.tv_sec * 1e6 + real_time.tv_nsec * 1e-3 - mono_time.tv_sec * 1e6 - mono_time.tv_nsec * 1e-3;

    return tsc * 32 / 1000 + diff_us;
}

SIPLStatus CVirtualCamProducer::GetPacketId(std::vector<NvSciBufObj> bufObjs, NvSciBufObj sciBufObj, uint32_t& packetId) {
    std::vector<NvSciBufObj>::iterator it = std::find_if(bufObjs.begin(), bufObjs.end(), [sciBufObj](const NvSciBufObj& obj) { return (sciBufObj == obj); });

    if (bufObjs.end() == it) {
        // Didn't find matching buffer
        PLOG_ERR("MapPayload, failed to get packet index for buffer\n");
        return NVSIPL_STATUS_ERROR;
    }

    packetId = std::distance(bufObjs.begin(), it);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::MapPayload(void* pBuffer, uint32_t& packetIndex) {
    // this buffer is h265 raw data.
    auto buf = *(reinterpret_cast<InputBuf*>(pBuffer));

    // *file_ << buf.data;

    NvSciBufObj obj = nullptr;
    if (!buffer_pool_->AquireBuffer((uint16_t*)&packetIndex, &obj)) {
        PLOG_ERR("AquireBuffer error!\n");
    }

    PLOG_DBG("AquireBuffer index=%d p=%p, buf.size = %d, buf.frame_type = %d!\n", packetIndex, obj, buf.data.size(), buf.frame_type);

    auto status = DecodeOnceSync(buf, obj);
    if (NVSIPL_STATUS_OK != status) {
        LOG_ERR("DecodeOnceSync status=%d\n", status);
        return NVSIPL_STATUS_ERROR;
    }

    // if (WriteBufferToFile(obj, "camera" + std::to_string(m_uSensorId) + ".yuv", FILE_IO_MODE_NVSCI)) {
    //     PLOG_ERR("WriteBufferToFile error!\n");
    // }

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());

    if ((uint64_t)duration.count() < buf.post_time) {
        auto wait = std::chrono::nanoseconds(buf.post_time) - duration;
        // PLOG_ERR("===wait %ums\n", wait.count() / 1000000);
        std::this_thread::sleep_for(wait);
    } else {
        PLOG_ERR("post_time before now!  %ldms\n", int64_t(buf.post_time - duration.count()) / 1000000);
        // TODO(mxt): if lost msg, sleep 5ms, then send.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // buffer_pool_->ReleaseBuffer(packetIndex);
        // return NVSIPL_STATUS_ERROR;
    }

    if (m_metaPtrs[packetIndex] != nullptr) {
        static_cast<MetaData*>(m_metaPtrs[packetIndex])->captureImgTimestamp = buf.post_time / 1000;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::CollectWaiterAttrList(uint32_t elementId, std::vector<NvSciSyncAttrList>& unreconciled) {
    SIPLStatus status = CProducer::CollectWaiterAttrList(elementId, unreconciled);
    PCHK_STATUS_AND_RETURN(status, "CollectWaiterAttrList");

    if (!m_spLateConsHelper || unreconciled.empty()) {
        return NVSIPL_STATUS_OK;
    }

    std::vector<NvSciSyncAttrList> lateAttrLists;
    status = m_spLateConsHelper->GetSyncWaiterAttrLists(lateAttrLists);
    if (status != NVSIPL_STATUS_OK) {
        PLOG_ERR("m_spLateConsHelper->GetSyncWaiterAttrList failed, status: %u\n", (status));
        return status;
    }
    unreconciled.insert(unreconciled.end(), lateAttrLists.begin(), lateAttrLists.end());
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::GetIdeImageAttributes(const NvSciBufAttrList& imageAttr) {
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
    ChromaFormat surfaceChromaFormat = YUV420SP_8bit;

    NvSciBufAttrValColorStd colorFormat = NvSciColorStd_REC709_ER;  // Initil  to avoid MISRA-C errors
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    auto status = NvMediaIDEFillNvSciBufAttrList(NVMEDIA_DECODER_INSTANCE_0, imageAttr);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate IDE internal attributes status=%d\n", status);
        return NVSIPL_STATUS_ERROR;
    }

    status = PopulateNvSciBufAttrListEx(surfaceChromaFormat, (pic_info_->width + 15) & ~15,     //
                                        (pic_info_->height + 15) & ~15, true,                   /* needCpuAccess */
                                        layout, IDE_APP_MAX_INPUT_PLANE_COUNT,                  //
                                        NvSciBufAccessPerm_ReadWrite, IDE_APP_BASE_ADDR_ALIGN,  //
                                        colorFormat, scanType, imageAttr);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate attributes\n");
        return NVSIPL_STATUS_ERROR;
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVirtualCamProducer::DecodeOnceSync(const InputBuf& src_data, NvSciBufObj out) {
    // COST_CALC_FUNC();

    NvMediaBitStreamPkt packet{0};
    packet.uDataLength = src_data.data.size();
    packet.pByteStream = (const uint8_t*)src_data.data.data();
    packet.bEOS = 0;
    packet.bPTSValid = 0;
    packet.llPts = 0;

    last_sci_buf_ = out;
    auto status = NvMediaParserParse(media_parser_, &packet);
    PCHK_NVMSTATUS_AND_RETURN(status, "NvMediaParserParse failed.");
    // NvMediaParserFlush(media_parser_);

    {
        std::unique_lock<std::mutex> lock(pic_info_h265_mutex_);
        auto pred = [this] {
            return pic_info_h265_parsed_;
        };

        if (!pic_info_h265_condition_.wait_for(lock, std::chrono::milliseconds(30), pred)) {
            PLOG_ERR("Wait media paser time out.\n");
            return NVSIPL_STATUS_ERROR;
        }
    }

    NvMediaIDEFrameStatus frame_status = {0};

    NvMediaBitstreamBuffer buffer{(uint8_t*)last_buf_.data(), last_buf_.size(), 0};

    // NvMediaBitstreamBuffer buffer{(uint8_t*)src_data.data.data(), src_data.data.size(), 0};

    // std::string file_name = std::string("camera_") + std::to_string(m_uSensorId) + "-vcp.265";
    // std::ofstream ofs(file_name, std::ios::binary | std::ios::app | std::ios::out);
    // ofs.write((const char*)(src_data.buf), src_data.size);

    PLOG_DBG("DecodeOnceSync size=%d, type=%d\n", src_data.data.size(), src_data.frame_type);

    // union {
    //     NvMediaPictureInfoH264 h264;
    //     NvMediaPictureInfoH265 hevc;
    // } pic_info;

    if (pic_info_->codec == NVMEDIA_VIDEO_CODEC_HEVC) {
        // TODO(zax): need set some h265 info??
        PLOG_DBG("width=%d, height=%d\n", pic_info_->width, pic_info_->height);
    }

    status = NvMediaIDEDecoderRender(decoder_,                     // decoder
                                     out,                          // target
                                     &pic_info_h265_,              // pictureInfo
                                     NULL,                         // encryptParams
                                     1,                            // numBitstreamBuffers
                                     &buffer,                      // bitstreams
                                     NULL,                         // FrameStatsDump
                                     NVMEDIA_DECODER_INSTANCE_0);  // instance ID

    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("cbDecodePicture: Decode failed: %d\n", status);
        return NVSIPL_STATUS_ERROR;
    }

    status = NvMediaIDEGetFrameDecodeStatus(decoder_, 0, &frame_status);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("cbDecodePicture: Frame Status decode_error %u\n", frame_status.decode_error);
    }

    NvSciSyncFence fence = NvSciSyncFenceInitializer;
    // 1->ELEMENT_TYPE_NV12_BL
    status = NvMediaIDEGetEOFNvSciSyncFence(decoder_, m_signalSyncObjs[1], &fence);
    PCHK_NVMSTATUS_AND_RETURN(status, "NvMediaIDEGetEOFNvSciSyncFence failed.");

    // COST_CALC_OP("wait fence");
    NvSciError nvSciErr = NvSciSyncFenceWait(&fence, m_cpuWaitContext, 1000 * 1000);
    PCHK_NVSCISTATUS_AND_RETURN(nvSciErr, "NvSciSyncFenceWait");

    // static int write_count = 0;
    // ++write_count;
    // std::string filename = std::string("camera_") + std::to_string(m_uSensorId) + "-" + std::to_string(write_count) + ".yuv";
    // WriteBufferToFile(out, filename, FILE_IO_MODE_NVSCI);
    return NVSIPL_STATUS_OK;
}

// SIPLStatus CVirtualCamProducer::Process(const InputBuf& buf) {
//     // TODO(zax): use first buffer only, optimize later.
//     PLOG_ERR("pOST  p=%p\n", m_vIspBufObjs[0]);
//     auto status = DecodeOnceSync({(uint8_t*)buf.buf, buf.size, 0}, &m_vIspBufObjs[0]);
//     if (NVSIPL_STATUS_OK != status) {
//         LOG_ERR("DecodeOnceSync status=%d\n", status);
//         return NVSIPL_STATUS_ERROR;
//     }
//     PLOG_ERR("pOST 2 p=%p\n", m_vIspBufObjs[0]);
//     return CProducer::Post(&m_vIspBufObjs[0]);
// }

NvMediaStatus CVirtualCamProducer::ParserCb_DecodePicture(void* ctx, NvMediaParserPictureData* pic_data) {
    CVirtualCamProducer* producer = static_cast<CVirtualCamProducer*>(ctx);
    {
        std::lock_guard<std::mutex> lock(producer->pic_info_h265_mutex_);
        producer->pic_info_h265_parsed_ = true;
        producer->pic_info_h265_ = pic_data->CodecSpecificInfo.hevc;
        // TODO(mxt): input h265 buffer size is not equal to out frame buffer frome media parser.
        // some extra data appended.
        // do convert in this scope.
        producer->last_buf_.clear();
        producer->last_buf_.append((char*)pic_data->pBitstreamData, pic_data->uBitstreamDataLen);
    }

    producer->pic_info_h265_condition_.notify_one();
    return NVMEDIA_STATUS_OK;
}

int32_t CVirtualCamProducer::ParserCb_BeginSequence(void* ctx, const NvMediaParserSeqInfo* pnvsi) {
    return pnvsi->uDecodeBuffers;
}

NvMediaStatus CVirtualCamProducer::ParserCb_DisplayPicture(void* ctx, NvMediaRefSurface* p_ref_surf, int64_t llpts) {
    return NVMEDIA_STATUS_OK;
}

void CVirtualCamProducer::ParserCb_UnhandledNALU(void* ctx, const uint8_t* buf, int32_t size) {}

NvMediaStatus CVirtualCamProducer::ParserCb_AllocPictureBuffer(void* ctx, NvMediaRefSurface** pp_ref_surf) {
    CVirtualCamProducer* producer = static_cast<CVirtualCamProducer*>(ctx);
    *pp_ref_surf = producer->last_sci_buf_;
    return NVMEDIA_STATUS_OK;
}

void CVirtualCamProducer::ParserCb_Release(void* ctx, NvMediaRefSurface* p_ref_surf) {}

void CVirtualCamProducer::ParserCb_AddRef(void* ctx, NvMediaRefSurface* p_ref_surf) {}

NvMediaStatus CVirtualCamProducer::ParserCb_GetBackwardUpdates(void* ptr, NvMediaVP9BackwardUpdates* backwardUpdate) {
    return NVMEDIA_STATUS_OK;
}