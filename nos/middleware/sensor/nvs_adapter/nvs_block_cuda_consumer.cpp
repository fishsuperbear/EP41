#include "sensor/nvs_adapter/nvs_block_cuda_consumer.h"
#include "sensor/nvs_adapter/nvs_helper.h"
#include <chrono>

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NVSBlockCUDAConsumer::Create(NvSciStreamBlock pool, const std::string& endpoint_info, PacketReadyCallback callback) {
    name = endpoint_info;

    NvSciError stream_err;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamConsumerCreate(pool, &block),
        "create consumer block")

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockUserInfoSet(block, ENDINFO_NAME_PROC, endpoint_info.size(), endpoint_info.c_str()),
        "setup consumer info")

    RegIntoEventService();

    _packet_ready_callback = callback;

    return 0;
}

int32_t NVSBlockCUDAConsumer::ReleasePacket(ImageCUDAPacket* packet) {
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

void NVSBlockCUDAConsumer::DeleteEventService() {
    NVSBlockCommon::DeleteEventService();
}

void NVSBlockCUDAConsumer::DeleteBlock() {
    NvSciStreamBlockDelete(block);

    for (auto& packet : _packets) {
        NvSciBufObjFree(packet->stream_buf_obj);
    }

    NvSciSyncObjFree(_signal_obj);
    NvSciSyncObjFree(_waiter_obj);
    
    NvSciSyncCpuWaitContextFree(_cpu_wait_context);
}

int32_t NVSBlockCUDAConsumer::OnConnected() {
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

int32_t NVSBlockCUDAConsumer::OnElements() {
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
                { NvSciBufImageAttrKey_Size, NULL, 0 },                     //0
                { NvSciBufImageAttrKey_Layout, NULL, 0 },                   //1
                { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },               //2
                { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },               //3
                { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },              //4
                { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },               //5
                { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },        //6
                { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 },       //7
                { NvSciBufImageAttrKey_PlaneOffset, NULL, 0 },              //8
                { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },         //9
            };

            NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListGetAttrs(buf_attr, keyVals, sizeof(keyVals) / sizeof(keyVals[0])),
                "obtain buffer size");
            _image_attrs.size = *((const uint64_t*)(keyVals[0].value));
            _image_attrs.layout = *((const NvSciBufAttrValImageLayoutType*)(keyVals[1].value));
            _image_attrs.plane_count = *((const uint32_t*)(keyVals[2].value));
            _image_attrs.width = *((const uint32_t*)(keyVals[3].value));
            _image_attrs.height = *((const uint32_t*)(keyVals[4].value));
            memcpy(_image_attrs.plane_widths, static_cast<const uint32_t *>(keyVals[3].value),
                _image_attrs.plane_count * sizeof(uint32_t));
            memcpy(_image_attrs.plane_heights, static_cast<const uint32_t *>(keyVals[4].value),
                _image_attrs.plane_count * sizeof(uint32_t));
            memcpy(_image_attrs.plane_pitches, static_cast<const uint32_t *>(keyVals[5].value),
                _image_attrs.plane_count * sizeof(uint32_t));
            memcpy(_image_attrs.plane_bits_per_pixels, static_cast<const uint32_t *>(keyVals[6].value),
                _image_attrs.plane_count * sizeof(uint32_t));
            memcpy(_image_attrs.plane_aligned_heights, static_cast<const uint32_t *>(keyVals[7].value),
                _image_attrs.plane_count * sizeof(uint32_t));
            memcpy(_image_attrs.plane_offsets, static_cast<const uint64_t *>(keyVals[8].value),
                _image_attrs.plane_count * sizeof(uint64_t));
            memcpy(_image_attrs.plane_color_formats, static_cast<const NvSciBufAttrValColorFmt *>(keyVals[9].value),
                _image_attrs.plane_count * sizeof(NvSciBufAttrValColorFmt));

            for (uint32_t i = 0; i < _image_attrs.plane_count; ++i) {
                _image_attrs.dst_size += _image_attrs.plane_widths[i] *
                             _image_attrs.plane_heights[i] * 
                             (_image_attrs.plane_bits_per_pixels[i] / 8);
            }
            NVS_LOG_INFO << "Dst data size " << _image_attrs.dst_size;

            // NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_Size: " << _image_attrs.data_size;
            // NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_Layout: " << _image_attrs.layout;
            // NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_PlaneCount: " << _image_attrs.plane_count;
            // NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_PlaneWidth0: " << _image_attrs.width;
            // NVS_LOG_INFO << "Src buf NvSciBufImageAttrKey_PlaneHeight0: " << _image_attrs.height;
            DumpBufAttrAll("IMAGE", buf_attr);
            NvSciBufAttrListFree(buf_attr);
            
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

int32_t NVSBlockCUDAConsumer::OnPacketCreate() {
    NvSciError stream_err;

    NVS_LOG_INFO << "Create NO." << _packets.size() << " packet.";

    // packet
    NvSciStreamPacket handle;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketNewHandleGet(block, &handle),
        "retrieve handle for the new packet");

    std::shared_ptr<ImageCUDAPacket> packet(new ImageCUDAPacket);
    _packets.emplace_back(packet);
    packet->handle = handle;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketBufferGet(block, handle, _image_ele_index, &packet->stream_buf_obj),
        "retrieve image buffer");

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

int32_t NVSBlockCUDAConsumer::OnPacketsComplete() {
    NvSciError stream_err;

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_PacketImport, true),
        "inform pool of packet status")
    return 0;
}

int32_t NVSBlockCUDAConsumer::OnWaiterAttr() {
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

int32_t NVSBlockCUDAConsumer::OnSignalObj() {
    NvSciError stream_err;

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementSignalObjGet(block, 0U, _image_ele_index, &_waiter_obj),
        "query sync object")

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockSetupStatusSet(block, NvSciStreamSetup_SignalObjImport, true),
        "complete signal obj import")

    return 0;
}

int32_t NVSBlockCUDAConsumer::ConsumerInit() {
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

    return 0;
}

int32_t NVSBlockCUDAConsumer::ConsumerElemSupport() {
    NvSciError stream_err;

    // image
    NvSciBufAttrList image_buf_attr;
    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &image_buf_attr),
        "create buff attr list")
    NVS_ASSERT_RETURN_INT(GetCUDABufAttr(image_buf_attr));
    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockElementAttrSet(block, ELEMENT_NAME_IMAGE, image_buf_attr),
        "send metadata attr")
    NvSciBufAttrListFree(image_buf_attr);

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

int32_t NVSBlockCUDAConsumer::ConsumerSyncSupport() {
    NvSciError stream_err;

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
    
    uint8_t waiter_cpu_sync = 1;
    NvSciSyncAccessPerm waiter_cpu_perm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair waiter_key_vals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &waiter_cpu_sync, sizeof(waiter_cpu_sync) },
        { NvSciSyncAttrKey_RequiredPerm,  &waiter_cpu_perm, sizeof(waiter_cpu_perm) }
    };
    NVS_ASSERT_RETURN_STREAM(NvSciSyncAttrListSetAttrs(_waiter_attr, waiter_key_vals, 2),
        "fill cpu signal sync attrs")
    
    // cpu wait
    NVS_ASSERT_RETURN_STREAM(NvSciSyncCpuWaitContextAlloc(NVSHelper::GetInstance().sci_sync_module, &_cpu_wait_context),
        "create CPU wait context")

    return 0;
}

int32_t NVSBlockCUDAConsumer::InitCUDA() {
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

int32_t NVSBlockCUDAConsumer::MapCUDAMem(ImageCUDAPacket* packet) {
    int32_t cuda_err;

    /* Map in the buffer as CUDA external memory */
    struct cudaExternalMemoryHandleDesc mem_handle_desc;
    memset(&mem_handle_desc, 0, sizeof(mem_handle_desc));
    mem_handle_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    mem_handle_desc.handle.nvSciBufObject = packet->stream_buf_obj;
    mem_handle_desc.size = _image_attrs.size;
    packet->data_size = _image_attrs.dst_size;
    packet->height = _image_attrs.height;
    packet->width = _image_attrs.width;

    NVS_ASSERT_RETURN_CUDA(cudaImportExternalMemory(&packet->cuda_ext_memory, &mem_handle_desc),
        "map buffer as external mem")

    if (_image_attrs.layout == NvSciBufImage_BlockLinearType) {
        cudaExtent extent[MAX_NUM_SURFACES];
        cudaChannelFormatDesc desc[MAX_NUM_SURFACES];
        cudaExternalMemoryMipmappedArrayDesc mipmapDesc[MAX_NUM_SURFACES];
        memset(extent, 0, MAX_NUM_SURFACES * sizeof(cudaExtent));
        memset(desc, 0, MAX_NUM_SURFACES * sizeof(cudaChannelFormatDesc));
        memset(mipmapDesc, 0, MAX_NUM_SURFACES * sizeof(cudaExternalMemoryMipmappedArrayDesc));

        for (uint32_t plane_id = 0; plane_id < _image_attrs.plane_count; plane_id++) {
            /* Setting for each plane buffer
             * SP format has 2 planes
             * Planar format has 3 planes  */

            // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaExtent.html#structcudaExtent
            // Width in elements when referring to array memory, in bytes when referring to linear memory
            // NvSciBufImageAttrKey_PlanePitch: Outputs the pitch (aka width in bytes) for every plane.
            // Bug 3880762
            extent[plane_id].width = _image_attrs.plane_pitches[plane_id] /
                                    (_image_attrs.plane_bits_per_pixels[plane_id] / 8);
            extent[plane_id].height = _image_attrs.plane_aligned_heights[plane_id];
            // Set the depth to 0 will create a 2D mapped array,
            // set the depth to 1 which indicates CUDA that it is a 3D array
            // Bug 3907432
            extent[plane_id].depth = 0;

            /* For Y */
            if (_image_attrs.plane_color_formats[plane_id] == NvSciColor_Y8) {
                desc[plane_id] = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
            }
            /* For UV */
            if ((_image_attrs.plane_color_formats[plane_id] == NvSciColor_U8V8) ||
                (_image_attrs.plane_color_formats[plane_id] == NvSciColor_U8_V8) ||
                (_image_attrs.plane_color_formats[plane_id] == NvSciColor_V8U8) ||
                (_image_attrs.plane_color_formats[plane_id] == NvSciColor_V8_U8)) {
                desc[plane_id] = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
            }
            mipmapDesc[plane_id].offset = _image_attrs.plane_offsets[plane_id];
            mipmapDesc[plane_id].formatDesc = desc[plane_id];
            mipmapDesc[plane_id].extent = extent[plane_id];
            mipmapDesc[plane_id].flags = 0;
            mipmapDesc[plane_id].numLevels = 1;
            NVS_ASSERT_RETURN_CUDA(
                cudaExternalMemoryGetMappedMipmappedArray(&packet->mipmapArray[plane_id], packet->cuda_ext_memory, &mipmapDesc[plane_id]),
                "get mapped mipmap array")

        } 
    }
    else if (_image_attrs.layout == NvSciBufImage_PitchLinearType) {
        /* Map in the buffer as CUDA device memory */
        struct cudaExternalMemoryBufferDesc mem_buffer_desc;
        memset(&mem_buffer_desc, 0, sizeof(mem_buffer_desc));
        mem_buffer_desc.size = _image_attrs.size;
        mem_buffer_desc.offset = 0;
        NVS_ASSERT_RETURN_CUDA(cudaExternalMemoryGetMappedBuffer(&packet->cuda_dev_ptr, packet->cuda_ext_memory, &mem_buffer_desc),
            "map buffer as device mem")
    }
    else {
        NVS_LOG_CRITICAL << "Unsupport layout.";
        return -1;
    }

    // /* Allocate normal memory to use as the target for the CUDA op */
    // packet->local_ptr = (uint8_t*)malloc(_image_attrs.dst_size);
    // if (NULL == packet->local_ptr) {
    //     NVS_LOG_CRITICAL << "Consumer failed to allocate target, ret " << log::loghex((uint32_t)cuda_err);
    //     return -5;
    // }

    // /* Fill in with initial values */
    // memset(packet->local_ptr, 0xD0, _image_attrs.dst_size);

    return 0;
}

int32_t NVSBlockCUDAConsumer::BlToPlConvert(ImageCUDAPacket* packet) {
    uint8_t *vaddr_plane[2] = { nullptr };
    int32_t cuda_err;
    
    NVS_ASSERT_RETURN_CUDA(cudaGetMipmappedArrayLevel(&packet->mipLevelArray[0U], packet->mipmapArray[0], 0U),
        "cudaGetMipmappedArrayLevel");
    NVS_ASSERT_RETURN_CUDA(cudaGetMipmappedArrayLevel(&packet->mipLevelArray[1U], packet->mipmapArray[1], 0U),
        "cudaGetMipmappedArrayLevel");

    vaddr_plane[0] = (uint8_t *)&packet->mipLevelArray[0U];
    vaddr_plane[1] = (uint8_t *)&packet->mipLevelArray[1U];

    packet->need_user_free = true;
    NVS_ASSERT_RETURN_CUDA(cudaMalloc((void **)&packet->cuda_dev_ptr, _image_attrs.dst_size),
        "malloc dst buffer");

    NVS_ASSERT_RETURN_CUDA(
        cudaMemcpy2DFromArray(packet->cuda_dev_ptr, _image_attrs.plane_widths[0], *((cudaArray_const_t *)vaddr_plane[0]), 0, 0,
            _image_attrs.plane_widths[0], _image_attrs.plane_heights[0], cudaMemcpyDeviceToDevice),
        "perform cuda memcpy plane 0")

    uint8_t* second_dst = (uint8_t *)packet->cuda_dev_ptr + _image_attrs.plane_widths[0] * _image_attrs.plane_heights[0];
    NVS_ASSERT_RETURN_CUDA(
        cudaMemcpy2DFromArray(second_dst, _image_attrs.plane_widths[0], *((cudaArray_const_t *)vaddr_plane[1]), 0, 0,
            _image_attrs.plane_widths[0], _image_attrs.plane_heights[0] / 2, cudaMemcpyDeviceToDevice),
        "perform cuda memcpy plane 1")

    return 0;
}

int32_t NVSBlockCUDAConsumer::GetCUDABufAttr(NvSciBufAttrList& attr) {
    NvSciError stream_err;

    NvSciBufAttrValAccessPerm dataPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType bufType = NvSciBufType_Image;
    bool cpu_access = true;
    // NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    // NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    // uint32_t planeCount = 1;
    // NvSciBufAttrValColorFmt colorFormat[] = { NvSciColor_Y8U8Y8V8 };    
    // uint32_t width[] = {_width};
    // uint32_t height[] = {_height};
    NvSciRmGpuId dataGpu = { 0 };
    memcpy(&dataGpu.bytes, &_cuda_uuid.bytes, sizeof(dataGpu.bytes));
    // uint32_t align = {1};
    NvSciBufAttrKeyValuePair keyVals[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpu_access, sizeof(cpu_access)},
        // { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        // { NvSciBufImageAttrKey_ScanType, &scanType, sizeof(scanType) },
        // { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount) },
        // { NvSciBufImageAttrKey_PlaneColorFormat, colorFormat, sizeof(colorFormat) },
        // { NvSciBufImageAttrKey_PlaneWidth, &width, sizeof(width) },
        // { NvSciBufImageAttrKey_PlaneHeight, &height, sizeof(height) },
        { NvSciBufGeneralAttrKey_GpuId, &dataGpu, sizeof(dataGpu)},
        { NvSciBufGeneralAttrKey_RequiredPerm, &dataPerm, sizeof(dataPerm)},
        // { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &align, sizeof(align)}
    };
    NVS_ASSERT_RETURN_STREAM(NvSciBufAttrListSetAttrs(attr, keyVals, sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair)), 
        "fill data attribute list")

    return 0;
}

int32_t NVSBlockCUDAConsumer::GetMetadataBufAttr(NvSciBufAttrList& attr) {
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

int32_t NVSBlockCUDAConsumer::OnPacketReady() {
    NvSciError stream_err;
    NvSciSyncFence fence = NvSciSyncFenceInitializer;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    /* Obtain packet with the new payload */
    NvSciStreamCookie cookie;
    NVS_ASSERT_RETURN_STREAM(NvSciStreamConsumerPacketAcquire(block, &cookie),
        "obtain packet for payload")

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    ImageCUDAPacket* packet = (ImageCUDAPacket*)cookie;

    NVS_ASSERT_RETURN_STREAM(NvSciStreamBlockPacketFenceGet(block, packet->handle, 0U, _image_ele_index, &fence),
        "query fence from producer");

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    NVS_ASSERT_RETURN_STREAM(NvSciSyncFenceWait(&fence, _cpu_wait_context, 0xFFFFFFFF),
        "wait for all operations done");

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (_image_attrs.layout == NvSciBufImage_BlockLinearType) {
        NVS_ASSERT_RETURN_INT(BlToPlConvert(packet));
    }
    else if (_image_attrs.layout == NvSciBufImage_PitchLinearType) {
        packet->need_user_free = false;
    }
    else {
        return -1;
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
    _packet_ready_callback(packet);
    std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();

    double d61 = std::chrono::duration<double, std::milli>(t6 - t1).count();
    if (d61 > 30) {
        double d21 = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double d32 = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double d43 = std::chrono::duration<double, std::milli>(t4 - t3).count();
        double d54 = std::chrono::duration<double, std::milli>(t5 - t4).count();
        double d65 = std::chrono::duration<double, std::milli>(t6 - t5).count();
        NVS_LOG_WARN << name << " PacketReady time cost, d61: " << d61 
            << ", d21: " << d21 
            << ", d32: " << d32 
            << ", d43: " << d43 
            << ", d54: " << d54 
            << ", d65: " << d65;
    }

    return 0;
}

}
}
}