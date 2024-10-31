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

struct ImageMediaCUDAPacket {
    NvSciStreamPacket handle;
    NvSciBufObj stream_buf_obj;

    NvSciBufObj metadata_buf_obj;
    SIPLImageMetadata* metadata_local_ptr;

    NvSciBufObj media_dst_buf_obj;

    cudaExternalMemory_t cuda_ext_memory;
    void* cuda_dev_ptr;
    uint8_t* local_ptr;

    uint64_t data_size;
};

class NVSBlockImgConsumer : public NVSBlockCommon {
public:
    using PacketReadyCallback = std::function<void(ImageMediaCUDAPacket*)>;

    int32_t Create(NvSciStreamBlock pool, const std::string& endpoint_info, PacketReadyCallback callback);

    int32_t ReleasePacket(ImageMediaCUDAPacket* packet);

protected:
    virtual void DeleteBlock() override;
    virtual int32_t OnConnected() override;
    virtual int32_t OnElements() override;
    virtual int32_t OnPacketCreate() override;
    virtual int32_t OnPacketsComplete() override;
    virtual int32_t OnWaiterAttr() override;
    virtual int32_t OnSignalObj() override;
    virtual int32_t OnPacketReady() override;

private:
    int32_t ConsumerInit();
    int32_t ConsumerElemSupport();
    int32_t ConsumerSyncSupport();

    int32_t InitCUDA();
    int32_t MapCUDAMem(ImageMediaCUDAPacket* packet);

    int32_t InitMedia2D();
    int32_t GetMedia2DSrcBufAttr(NvSciBufAttrList& attr);
    int32_t GetMedia2DDstCUDABufAttr(NvSciBufAttrList& attr);
    int32_t Media2DPerformCompose(ImageMediaCUDAPacket* packet);

    int32_t GetMetadataBufAttr(NvSciBufAttrList& attr);

    NvSciSyncAttrList _signal_attr;
    NvSciSyncAttrList _waiter_attr;
    NvSciSyncAttrList _media2d_eof_waiter_attr;
    NvSciSyncObj _signal_obj;
    NvSciSyncObj _waiter_obj;
    NvSciSyncObj _media2d_eof_waiter_obj;
    uint64_t _data_size;
    uint32_t _width;
    uint32_t _height;
    std::vector<std::shared_ptr<ImageMediaCUDAPacket>> _packets;
    NvSciSyncCpuWaitContext _cpu_wait_context;
    cudaStream_t _cuda_stream;
    CUuuid _cuda_uuid;
    PacketReadyCallback _packet_ready_callback;
    uint32_t _image_ele_index;
    uint32_t _metadata_ele_index;

    // media2d
    NvMedia2D* _media2d_handle;
    NvSciBufAttrList _media2d_dst_cuda_buf_attr;
};

}
}
}