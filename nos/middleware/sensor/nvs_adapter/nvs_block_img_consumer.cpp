#include "sensor/nvs_adapter/nvs_block_img_consumer.h"
#include "sensor/nvs_adapter/nvs_helper.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NVSBlockImgConsumer::Create(NvSciStreamBlock pool, const std::string& endpoint_info, PacketReadyCallback callback) {
    name = "CONSUMER";

    NvSciError stream_err;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamConsumerCreate(pool, &block),
        "create consumer block")

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockUserInfoSet(block, ENDINFO_NAME_PROC, endpoint_info.size(), endpoint_info.c_str()),
        "setup consumer info")

    RegIntoEventService();

    _packet_ready_callback = callback;

    return 0;
}

int32_t NVSBlockImgConsumer::ReleasePacket(ImageMediaCUDAPacket* packet) {
    NvSciError stream_err;
    NvSciSyncFence fence = NvSciSyncFenceInitializer;
    
    NVS_LOG_TRACE << "Release nvs packet."; 
    NVS_ASSERT_RETURN_STREAM(NvSciSyncObjGenerateFence(_signal_obj, &fence),
        "genreate fence")

    /* Update postfence for this element */
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketFenceSet(block, packet->handle, _image_ele_index, &fence),
        "set packet fence");

    NVS_ASSERT_RETURN_STREAM(NvSciStreamConsumerPacketRelease(block, packet->handle),
        "release packet")

    NvSciSyncObjSignal(_signal_obj);
    NvSciSyncFenceClear(&fence);

    return 0;
}

void NVSBlockImgConsumer::DeleteBlock() {
    NvSciStreamBlockDelete(block);

    for (auto& packet : _packets) {
        NvSciBufObjFree(packet->stream_buf_obj);
    }

    NvSciSyncObjFree(_signal_obj);
    NvSciSyncObjFree(_waiter_obj);
    
    NvSciSyncCpuWaitContextFree(_cpu_wait_context);
}

int32_t NVSBlockImgConsumer::OnConnected() {
    if (ConsumerInit() < 0) {
        return -1;
    }

    if (ConsumerElemSupport() < 0) {
        return -2;
    }

    if (ConsumerSyncSupport() < 0) {
        return -3;
    }

    return 0;
}

int32_t NVSBlockImgConsumer::OnElements() {
    NvSciError stream_err;

    uint32_t count;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementCountGet(block, NvSciStreamBlockType_Pool, &count),
        "query element count")
    if (2 != count) {
        NVS_LOG_CRITICAL << "Consumer received unexpected element count " << count;
        return -2;
    }
    NVS_LOG_INFO << "Consumer received element count " << count;

    for (int32_t i = 0; i < 2; ++i) {
        uint32_t type;
        NvSciBufAttrList buf_attr;
        NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementAttrGet(block, NvSciStreamBlockType_Pool, i, &type, &buf_attr),
            "query element attr");

        if (type == ELEMENT_NAME_IMAGE) {
            _image_ele_index = i;

            NvSciBufAttrKeyValuePair keyVals[] = {
                {NvSciBufImageAttrKey_Size, NULL, 0},
                {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},
                {NvSciBufImageAttrKey_PlaneHeight, NULL, 0}
            };
            NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListGetAttrs(buf_attr, keyVals, 3),
                "obtain buffer size");
            uint64_t src_data_size = *((const uint64_t*)(keyVals[0].value));
            _width = *((const uint32_t*)(keyVals[1].value));
            _height = *((const uint32_t*)(keyVals[2].value));
            NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_Size: " << src_data_size;
            NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_PlaneWidth: " << _width;
            NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_PlaneHeight: " << _height;
            NvSciBufAttrListFree(buf_attr);

            NVS_ASSERT_RETURN_INT(GetMedia2DDstCUDABufAttr(_media2d_dst_cuda_buf_attr))
            // DumpBufAttrAll(_media2d_dst_cuda_buf_attr);
            
            NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementWaiterAttrSet(block, _image_ele_index, _waiter_attr),
                "send waiter attrs");
            NvSciSyncAttrListFree(_waiter_attr);
            _waiter_attr = NULL;
        }
        else if (type == ELEMENT_NAME_METADATA) {
            _metadata_ele_index = i;
        }
        else {
            NVS_LOG_ERROR << "Received unknown element type " << log::loghex(type);
        }

        NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementUsageSet(block, i, true),
            "indicate element is used");
    }


    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_ElementImport, true),
        "complete element import")

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_WaiterAttrExport, true),
        "complete waiter attrs export")

    return 0;
}

int32_t NVSBlockImgConsumer::OnPacketCreate() {
    NvSciError stream_err;
    NvMediaStatus media_err;

    NVS_LOG_INFO << "Create NO." << _packets.size() << " packet.";

    // packet
    NvSciStreamPacket handle;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketNewHandleGet(block, &handle),
        "retrieve handle for the new packet");

    std::shared_ptr<ImageMediaCUDAPacket> packet(new ImageMediaCUDAPacket);
    _packets.emplace_back(packet);
    packet->handle = handle;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketBufferGet(block, handle, _image_ele_index, &packet->stream_buf_obj),
        "retrieve image buffer");

    // NVS_LOG_INFO << "bufobj " << packet->stream_buf_obj;
    // DumpBufAttrAll(packet->stream_buf_obj);
    NVS_ASSERT_RETURN_MEDIA(NvMedia2DRegisterNvSciBufObj(_media2d_handle, packet->stream_buf_obj),
        "register src buffer to nvmedia")

    // alloc dst obj
    // DumpBufAttrAll(_media2d_dst_cuda_buf_attr);
    NVS_ASSERT_RETURN_STREAM(NvSciBufObjAlloc(_media2d_dst_cuda_buf_attr, &packet->media_dst_buf_obj),
        "allocate media2d dst obj")
    NVS_ASSERT_RETURN_MEDIA(NvMedia2DRegisterNvSciBufObj(_media2d_handle, packet->media_dst_buf_obj),
        "register dst buffer to nvmedia")
    if (_packets.size() == 1) {
        NVS_LOG_INFO << "--------------- SRC OBJ ATTR ---------------";
        DumpBufAttrAll(packet->stream_buf_obj);
        NVS_LOG_INFO << "--------------------------------------------";
        NVS_LOG_INFO << "--------------- DST OBJ ATTR ---------------";
        DumpBufAttrAll(packet->media_dst_buf_obj);
        NVS_LOG_INFO << "--------------------------------------------";
    }

    // map to CUDA
    NVS_ASSERT_RETURN_INT(MapCUDAMem(packet.get()))

    // metadata
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketBufferGet(block, handle, _metadata_ele_index, &packet->metadata_buf_obj),
        "retrieve metadata buffer")

    NVS_ASSERT_RETURN_STREAM(NvSciBufObjGetConstCpuPtr(packet->metadata_buf_obj, (void const**)&packet->metadata_local_ptr),
        "map metadata buffe")

    // inform
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketStatusSet(block, handle, (NvSciStreamCookie)(packet.get()), NvSciError_Success),
        "inform pool of packet status")

    return 0;
}

int32_t NVSBlockImgConsumer::OnPacketsComplete() {
    NvSciError stream_err;

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_PacketImport, true),
        "inform pool of packet status")
    return 0;
}

int32_t NVSBlockImgConsumer::OnWaiterAttr() {
    NvSciError stream_err;

    NvSciSyncAttrList prod_waiter_attr;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementWaiterAttrGet(block, _image_ele_index, &prod_waiter_attr),
        "to query waiter attr");
    if (NULL == prod_waiter_attr) {
        NVS_LOG_CRITICAL << "Consumer received NULL waiter attr for data elem";
        return -2;
    }

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_WaiterAttrImport, true),
        "complete waiter attr import")

    NvSciSyncAttrList unreconciled[2] = {
        _signal_attr,
        prod_waiter_attr
    };
    NvSciSyncAttrList reconciled = NULL;
    NvSciSyncAttrList conflicts = NULL;
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListReconcile(unreconciled, 2, &reconciled, &conflicts),
        "reconcile sync attributes")

    NVS_ASSERT_RETURN_STREAM(NvSciSyncObjAlloc(reconciled, &_signal_obj),
        "allocate sync object")
        
    NvSciSyncAttrListFree(_signal_attr);
    _signal_attr = NULL;
    NvSciSyncAttrListFree(prod_waiter_attr);
    NvSciSyncAttrListFree(reconciled);

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementSignalObjSet(block, _image_ele_index, _signal_obj),
        "send sync object")

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_SignalObjExport, true),
        "complete signal obj export")

    return 0;
}

int32_t NVSBlockImgConsumer::OnSignalObj() {
    NvSciError stream_err;
    NvMediaStatus media_err;

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementSignalObjGet(block, 0U, _image_ele_index, &_waiter_obj),
        "query sync object")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DRegisterNvSciSyncObj(_media2d_handle, NVMEDIA_PRESYNCOBJ, _waiter_obj),
        "register media2d presync obj")

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_SignalObjImport, true),
        "complete signal obj import")

    return 0;
}

int32_t NVSBlockImgConsumer::ConsumerInit() {
    /* Query endpoint info from producer */
    uint32_t size = 50;
    char info[50];
    NvSciError err = NvSciStreamBlockUserInfoGet(
                        block,
                        NvSciStreamBlockType_Producer, 0U,
                        ENDINFO_NAME_PROC,
                        &size, &info);
    if (NvSciError_Success == err) {
        NVS_LOG_INFO << "Producer info: " << info;
    } 
    else if (NvSciError_StreamInfoNotProvided == err) {
        NVS_LOG_WARN << "Info not provided by the producer";
    } 
    else {
        NVS_LOG_ERROR << "Failed (%x) to query the producer info, ret " << LogHexNvErr(err);
    }

    NVS_ASSERT_RETURN_INT(InitCUDA())
    NVS_ASSERT_RETURN_INT(InitMedia2D())

    return 0;
}

int32_t NVSBlockImgConsumer::ConsumerElemSupport() {
    NvSciError stream_err;

    // image
    NvSciBufAttrList image_buf_attr;
    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &image_buf_attr),
        "create buff attr list")
    NVS_ASSERT_RETURN_INT(GetMedia2DSrcBufAttr(image_buf_attr));
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementAttrSet(block, ELEMENT_NAME_IMAGE, image_buf_attr),
        "send metadata attr")
    NvSciBufAttrListFree(image_buf_attr);

    // NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &_media2d_dst_cuda_buf_attr),
    //     "create dst buff attr list")
    // NVS_ASSERT_RETURN_INT(GetMedia2DDstCUDABufAttr(_media2d_dst_cuda_buf_attr))

    // metadata
    NvSciBufAttrList metadata_buf_attr;
    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &metadata_buf_attr),
        "create metadata buff attr list")
    NVS_ASSERT_RETURN_INT(GetMetadataBufAttr(metadata_buf_attr))
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementAttrSet(block, ELEMENT_NAME_METADATA, metadata_buf_attr),
        "send metadata attr")
    NvSciBufAttrListFree(metadata_buf_attr);

    /* Indicate that all element information has been exported */
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_ElementExport, true),
        "complete element export")

    return 0;
}

int32_t NVSBlockImgConsumer::ConsumerSyncSupport() {
    NvSciError stream_err;
    NvMediaStatus media_err;

    // signal
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListCreate(NVSHelper::GetInstance().sci_sync_module, &_signal_attr),
        "allocate signal sync attrs")

    uint8_t cpu_sync = 1;
    NvSciSyncAccessPerm cpu_perm = NvSciSyncAccessPerm_SignalOnly;
    NvSciSyncAttrKeyValuePair cpu_key_vals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpu_sync, sizeof(cpu_sync) },
        { NvSciSyncAttrKey_RequiredPerm,  &cpu_perm, sizeof(cpu_perm) }
    };
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListSetAttrs(_signal_attr, cpu_key_vals, 2),
        "fill cpu signal sync attrs")

    // waiter
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListCreate(NVSHelper::GetInstance().sci_sync_module, &_waiter_attr),
        "allocate waiter sync attrs")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DFillNvSciSyncAttrList(_media2d_handle, _waiter_attr, NVMEDIA_WAITER),
        "fill media2d waiter attrs")

    // media2d eof waiter
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListCreate(NVSHelper::GetInstance().sci_sync_module, &_media2d_eof_waiter_attr),
        "allocate media2d signaler sync attrs")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DFillNvSciSyncAttrList(_media2d_handle, _media2d_eof_waiter_attr, NVMEDIA_SIGNALER),
        "fill media2d signaler attrs")

    NvSciSyncAttrList cpu_wait_attr;
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListCreate(NVSHelper::GetInstance().sci_sync_module, &cpu_wait_attr),
        "allocate cpu wait sync attrs")
    
    uint8_t waiter_cpu_sync = 1;
    NvSciSyncAccessPerm waiter_cpu_perm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair waiter_cpu_key_vals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &waiter_cpu_sync, sizeof(waiter_cpu_sync) },
        { NvSciSyncAttrKey_RequiredPerm,  &waiter_cpu_perm, sizeof(waiter_cpu_perm) }
    };
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListSetAttrs(cpu_wait_attr, waiter_cpu_key_vals, 2),
        "fill cpu signal sync attrs")

    NvSciSyncAttrList unreconciled[2] = {
        _media2d_eof_waiter_attr,
        cpu_wait_attr
    };
    NvSciSyncAttrList reconciled = NULL;
    NvSciSyncAttrList conflicts = NULL;
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListReconcile(unreconciled, 2, &reconciled, &conflicts),
        "reconcile media2d eof sync attributes")
        
    NVS_ASSERT_RETURN_STREAM(NvSciSyncObjAlloc(reconciled, &_media2d_eof_waiter_obj),
        "allocate sync object")
    
    NVS_ASSERT_RETURN_MEDIA(NvMedia2DRegisterNvSciSyncObj(_media2d_handle, NVMEDIA_EOFSYNCOBJ, _media2d_eof_waiter_obj),
        "register media2d eofsync obj")

    NvSciSyncAttrListFree(_media2d_eof_waiter_attr);
    _media2d_eof_waiter_attr = NULL;
    NvSciSyncAttrListFree(cpu_wait_attr);
    NvSciSyncAttrListFree(reconciled);
    
    // cpu wait
    NVS_ASSERT_RETURN_STREAM(NvSciSyncCpuWaitContextAlloc(NVSHelper::GetInstance().sci_sync_module, &_cpu_wait_context),
        "create CPU wait context")

    return 0;
}

int32_t NVSBlockImgConsumer::InitCUDA() {
    size_t unused;
    int32_t cudart_err = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    if (cudaSuccess != cudart_err) {
        NVS_LOG_CRITICAL << "Fail to get CUDA device limit, ret " << cudart_err;
        return 0;
    }

    cudart_err = cudaSetDevice(0);
    if (cudaSuccess != cudart_err) {
        NVS_LOG_CRITICAL << "Fail to set cuda device, ret " << cudart_err;
        return -1;
    }

    CUresult cuda_err = cuDeviceGetUuid(&_cuda_uuid, 0);
    if (CUDA_SUCCESS != cuda_err) {
        NVS_LOG_CRITICAL << "Fail to get cuda uuid, ret " << cudart_err;
        return -1;
    }

    return 0;
}

int32_t NVSBlockImgConsumer::MapCUDAMem(ImageMediaCUDAPacket* packet) {
    int32_t cuda_err;

    /* Map in the buffer as CUDA external memory */
    struct cudaExternalMemoryHandleDesc mem_handle_desc;
    memset(&mem_handle_desc, 0, sizeof(mem_handle_desc));
    mem_handle_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    mem_handle_desc.handle.nvSciBufObject = packet->media_dst_buf_obj;
    mem_handle_desc.size = _data_size;
    packet->data_size = _data_size;
    // NVS_LOG_DEBUG << "nvSciBufObject: " << mem_handle_desc.handle.nvSciBufObject << ", size: " << mem_handle_desc.size;
    NVS_ASSERT_RETURN_CUDA(cudaImportExternalMemory(&packet->cuda_ext_memory, &mem_handle_desc),
        "map buffer as external mem")

    /* Map in the buffer as CUDA device memory */
    struct cudaExternalMemoryBufferDesc mem_buffer_desc;
    memset(&mem_buffer_desc, 0, sizeof(mem_buffer_desc));
    mem_buffer_desc.size = _data_size;
    mem_buffer_desc.offset = 0;
    NVS_ASSERT_RETURN_CUDA(cudaExternalMemoryGetMappedBuffer(&packet->cuda_dev_ptr, packet->cuda_ext_memory, &mem_buffer_desc),
        "map buffer as device mem")

    /* Allocate normal memory to use as the target for the CUDA op */
    packet->local_ptr = (uint8_t*)malloc(_data_size);
    if (NULL == packet->local_ptr) {
        NVS_LOG_CRITICAL << "Consumer failed to allocate target, ret " << log::loghex((uint32_t)cuda_err);
        return -5;
    }

    /* Fill in with initial values */
    memset(packet->local_ptr, 0xD0, _data_size);

    return 0;
}

int32_t NVSBlockImgConsumer::InitMedia2D() {
    NvMediaStatus err;
    err = NvMedia2DCreate(&_media2d_handle, NULL);
    if (err != NVMEDIA_STATUS_OK) {
        NVS_LOG_CRITICAL << "Fail to create media2d, ret " << err;
        return -1;
    }

    return 0;
}

int32_t NVSBlockImgConsumer::GetMedia2DSrcBufAttr(NvSciBufAttrList& attr) {
    NvSciError stream_err;
    NvMediaStatus media_err;

    // NvSciBufAttrList media2d_attr;
    // NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &media2d_attr),
    //     "create metadata buff attr list")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DFillNvSciBufAttrList(_media2d_handle, attr),
        "fill media2d buff attr");

    uint8_t dataCpu = 1U;
    NvSciBufAttrValAccessPerm dataPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType dataBufType = NvSciBufType_Image;
    // uint64_t align = 1;
    NvSciBufAttrKeyValuePair dataKeyVals[] = {
        {NvSciBufGeneralAttrKey_Types, &dataBufType, sizeof(dataBufType)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &dataPerm, sizeof(dataPerm)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &dataCpu, sizeof(dataCpu)},
        // {NvSciBufImageAttrKey_Alignment, &align, sizeof(align)}
    };

    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListSetAttrs(attr, dataKeyVals, sizeof(dataKeyVals) / sizeof(NvSciBufAttrKeyValuePair)), 
        "fill data attribute list")

    return 0;
}

int32_t NVSBlockImgConsumer::GetMedia2DDstCUDABufAttr(NvSciBufAttrList& attr) {
    NvSciError stream_err;
    NvMediaStatus media_err;
    
    // media2d
    NvSciBufAttrList media2d_attr;
    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &media2d_attr),
        "create metadata buff attr list")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DFillNvSciBufAttrList(_media2d_handle, media2d_attr),
        "fill media2d buff attr");

    NvSciBufAttrValAccessPerm dataPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType bufType = NvSciBufType_Image;
    bool cpu_access = true;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    uint32_t planeCount = 1;
    NvSciBufAttrValColorFmt colorFormat[] = { NvSciColor_Y8U8Y8V8 };    
    uint32_t width[] = {_width};
    uint32_t height[] = {_height};
    NvSciRmGpuId dataGpu = { 0 };
    memcpy(&dataGpu.bytes, &_cuda_uuid.bytes, sizeof(dataGpu.bytes));
    // uint32_t align = {1};
    NvSciBufAttrKeyValuePair keyVals[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpu_access, sizeof(cpu_access)},
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_ScanType, &scanType, sizeof(scanType) },
        { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount) },
        { NvSciBufImageAttrKey_PlaneColorFormat, colorFormat, sizeof(colorFormat) },
        { NvSciBufImageAttrKey_PlaneWidth, &width, sizeof(width) },
        { NvSciBufImageAttrKey_PlaneHeight, &height, sizeof(height) },
        { NvSciBufGeneralAttrKey_GpuId, &dataGpu, sizeof(dataGpu)},
        { NvSciBufGeneralAttrKey_RequiredPerm, &dataPerm, sizeof(dataPerm)},
        // { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &align, sizeof(align)}
    };
    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListSetAttrs(media2d_attr, keyVals, sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair)), 
        "fill data attribute list")

    NvSciBufAttrList attr_lists[] = {media2d_attr};
    NvSciBufAttrList conflicts = NULL;
    NvSciBufAttrList reconciled = NULL;
    // DumpBufAttrAll(media2d_attr);

    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListReconcile(attr_lists, 1, &reconciled, &conflicts),
        "reconcile cuda attr and media2d attr")

    attr = reconciled;
    // NVS_LOG_INFO << "--------------- MEDIA2D ATTR ---------------";
    // DumpBufAttrAll(media2d_attr);
    // NVS_LOG_INFO << "--------------------------------------------";
    // NVS_LOG_INFO << "--------------- MEDIA RECON ATTR ---------------";
    // DumpBufAttrAll(attr);
    // NVS_LOG_INFO << "--------------------------------------------";
    NvSciBufAttrKeyValuePair query_key_vals[] = {
        {NvSciBufImageAttrKey_Size, NULL, 0},
    };
    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListGetAttrs(attr, query_key_vals, 1),
        "query dst buff size");
    _data_size = *(uint64_t*)(query_key_vals[0].value);

    return 0;
}

int32_t NVSBlockImgConsumer::Media2DPerformCompose(ImageMediaCUDAPacket* packet) {
    NvSciError stream_err;
    NvMediaStatus media_err;
    NvSciSyncFence prefence = NvSciSyncFenceInitializer;
    NvSciSyncFence eoffence = NvSciSyncFenceInitializer;
    NvMedia2DComposeParameters params = 0;

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketFenceGet(block, packet->handle, 0U,  _image_ele_index, &prefence),
        "get packet fence");

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DGetComposeParameters(_media2d_handle, &params),
        "get media2d parameters")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DSetNvSciSyncObjforEOF(_media2d_handle, params, _media2d_eof_waiter_obj),
        "set media2d eof obj")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DSetSrcNvSciBufObj(_media2d_handle, params, 0, packet->stream_buf_obj),
        "set media2d src obj")

    NVS_ASSERT_RETURN_MEDIA(NvMedia2DSetDstNvSciBufObj(_media2d_handle, params, packet->media_dst_buf_obj),
        "set media2d dst obj")

    NvMedia2DComposeResult compose_result;
    NVS_ASSERT_RETURN_MEDIA(NvMedia2DCompose(_media2d_handle, params, &compose_result),
        "perform compose")
    
    NVS_ASSERT_RETURN_MEDIA(NvMedia2DGetEOFNvSciSyncFence(_media2d_handle, &compose_result, &eoffence),
        "get eof fence")

    NVS_ASSERT_RETURN_STREAM(NvSciSyncFenceWait(&eoffence, _cpu_wait_context, -1),
        "wait eof fence")

    NvSciSyncFenceClear(&eoffence);
    
    return 0;
}

int32_t NVSBlockImgConsumer::GetMetadataBufAttr(NvSciBufAttrList& attr) {
    NvSciError stream_err;

    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    uint8_t cpu_access = 1U;
    NvSciBufType buf_type = NvSciBufType_RawBuffer;
    uint64_t size = sizeof(SIPLImageMetadata);
    uint64_t align = 1U;
    NvSciBufAttrKeyValuePair key_vals[] = {
        { NvSciBufGeneralAttrKey_Types, &buf_type, sizeof(buf_type) },
        { NvSciBufRawBufferAttrKey_Size, &size, sizeof(size) },
        { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpu_access, sizeof(cpu_access) }
    };

    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListSetAttrs(attr, key_vals, sizeof(key_vals) / sizeof(NvSciBufAttrKeyValuePair)),
        "fill metadata attribute list");

    return 0;
}

int32_t NVSBlockImgConsumer::OnPacketReady() {
    NvSciError stream_err;

    /* Obtain packet with the new payload */
    NvSciStreamCookie cookie;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamConsumerPacketAcquire(block, &cookie),
        "obtain packet for payload")

    ImageMediaCUDAPacket* packet = (ImageMediaCUDAPacket*)cookie;

    NVS_ASSERT_RETURN_INT(Media2DPerformCompose(packet))

    _packet_ready_callback(packet);

    return 0;
}

}
}
}