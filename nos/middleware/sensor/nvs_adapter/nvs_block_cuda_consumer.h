#pragma once

#include "sensor/nvs_adapter/nvs_block_common.h"
#include "sensor/nvs_adapter/nvs_block_sipl_producer.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvmedia_2d.h"
#include "nvmedia_2d_sci.h"
#include <string>
#include <vector>
#include <functional>

namespace hozon {
namespace netaos {
namespace nv { 

#define MAX_NUM_SURFACES (3U)

struct ImageCUDAPacket {
    NvSciStreamPacket handle;
    NvSciBufObj stream_buf_obj;

    NvSciBufObj metadata_buf_obj;
    SIPLImageMetadata* metadata_local_ptr;

    cudaExternalMemory_t cuda_ext_memory;
    void* cuda_dev_ptr;

    uint64_t data_size;
    uint64_t width;
    uint64_t height;

    cudaMipmappedArray_t mipmapArray[MAX_NUM_SURFACES];
    cudaArray_t mipLevelArray[MAX_NUM_SURFACES];

    bool need_user_free;
};

class NVSBlockCUDAConsumer : public NVSBlockCommon {
public:
    using PacketReadyCallback = std::function<void(ImageCUDAPacket*)>;

    int32_t Create(NvSciStreamBlock pool, const std::string& endpoint_info, PacketReadyCallback callback);

    int32_t ReleasePacket(ImageCUDAPacket* packet);

    virtual void DeleteBlock() override;
    virtual void DeleteEventService() override;

protected:    
    virtual int32_t OnConnected() override;
    virtual int32_t OnElements() override;
    virtual int32_t OnPacketCreate() override;
    virtual int32_t OnPacketsComplete() override;
    virtual int32_t OnWaiterAttr() override;
    virtual int32_t OnSignalObj() override;
    virtual int32_t OnPacketReady() override;

private:
    struct ImageAttrs {
        uint64_t size;
        uint64_t dst_size;
        uint32_t width;
        uint32_t height;
        NvSciBufAttrValImageLayoutType layout;
        uint32_t plane_count;
        uint32_t plane_widths[MAX_NUM_SURFACES];
        uint32_t plane_heights[MAX_NUM_SURFACES];
        uint32_t plane_pitches[MAX_NUM_SURFACES];
        uint32_t plane_bits_per_pixels[MAX_NUM_SURFACES];
        uint32_t plane_aligned_heights[MAX_NUM_SURFACES];
        uint64_t plane_offsets[MAX_NUM_SURFACES];
        NvSciBufAttrValColorFmt plane_color_formats[MAX_NUM_SURFACES];
    };

    int32_t ConsumerInit();
    int32_t ConsumerElemSupport();
    int32_t ConsumerSyncSupport();

    int32_t InitCUDA();
    int32_t MapCUDAMem(ImageCUDAPacket* packet);
    int32_t GetCUDABufAttr(NvSciBufAttrList& attr);
    int32_t BlToPlConvert(ImageCUDAPacket* packet);

    int32_t GetMetadataBufAttr(NvSciBufAttrList& attr);

    NvSciSyncAttrList _signal_attr;
    NvSciSyncAttrList _waiter_attr;
    NvSciSyncObj _signal_obj;
    NvSciSyncObj _waiter_obj;
    ImageAttrs _image_attrs;
    std::vector<std::shared_ptr<ImageCUDAPacket>> _packets;
    NvSciSyncCpuWaitContext _cpu_wait_context;
    cudaStream_t _cuda_stream;
    CUuuid _cuda_uuid;
    PacketReadyCallback _packet_ready_callback;
    uint32_t _image_ele_index;
    uint32_t _metadata_ele_index;
};

}
}
}