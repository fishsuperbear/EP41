#pragma once

#include "sensor/nvs_adapter/nvs_block_common.h"
#include <vector>
#include <NvSIPLClient.hpp>
#include <NvSIPLCamera.hpp>

namespace hozon {
namespace netaos {
namespace nv { 

using namespace nvsipl;

#define SIPL_IMAGE_METADATA_VERSION     0x00000001

struct SIPLImageMetadata {
    uint32_t version;
    uint64_t capture_start_us;
    uint64_t capture_end_us;
};

struct SIPLProducerPacket {
    NvSciStreamPacket handle;
    NvSciBufObj image_buf_obj;
    NvSciBufObj metadata_buf_obj;
    SIPLImageMetadata* metadata_cpu_ptr;
    INvSIPLClient::INvSIPLNvMBuffer* sipl_buffer = nullptr;
};


class NVSBlockSIPLProducer : public NVSBlockCommon {
public:
    int32_t Create(NvSciStreamBlock pool, 
        const std::string& endpoint_info, 
        INvSIPLClient::ConsumerDesc::OutputType output_type, 
        uint32_t sensor_id,
        INvSIPLCamera* sipl_camera);

    int32_t Post(INvSIPLClient::INvSIPLNvMBuffer* sipl_buffer);
    bool Ready();
    
protected:
    virtual int32_t OnConnected() override;
    virtual int32_t OnElements() override;
    virtual int32_t OnPacketCreate() override;
    virtual int32_t OnPacketsComplete() override;
    virtual int32_t OnWaiterAttr() override;
    virtual int32_t OnSignalObj() override;
    virtual int32_t OnPacketReady() override;
    virtual int32_t OnSetupComplete() override;

private:
    int32_t StreamInit();
    int32_t ProducerInit();
    int32_t ProducerElemSupport();
    int32_t ProducerSyncSupport();
    void PrintImageAttr(NvSciBufAttrList& attr, const std::string& head);

    int32_t SIPLGetImageBufAttr(NvSciBufAttrList& buf_attr, INvSIPLClient::ConsumerDesc::OutputType output_type);
    int32_t SIPLGetSyncAttr();
    int32_t SIPLRegisterImages(std::vector<NvSciBufObj>& buf_objs, INvSIPLClient::ConsumerDesc::OutputType output_type);
    int32_t SIPLRegisterSignalObj();
    int32_t SIPLRegisterWaiterObj(uint32_t index);
    int32_t SIPLWaitPreFence(SIPLProducerPacket* packet, NvSciSyncFence& fence);
    int32_t SIPLGetMetadataAttr();
    int32_t SIPLRegisterExtraICPBuf();
    int32_t SIPLOverrideISPAttr(NvSciBufAttrList& attr);

    uint32_t _sensor_id;
    INvSIPLClient::ConsumerDesc::OutputType _output_type;
    INvSIPLCamera* _sipl_camera;
    NvSciBufAttrList _sipl_extra_icp_buf_attr;
    NvSciBufAttrList _metadata_buf_attr;
    NvSciBufAttrList _sipl_buf_attr;
    uint32_t _num_consumers;
    NvSciSyncAttrList _signal_attr;
    NvSciSyncAttrList _waiter_attr;
    NvSciSyncObj _signal_obj;
    NvSciSyncObj _waiter_obj[MAX_CONSUMERS];
    NvSciSyncCpuWaitContext _cpu_wait_context;
    std::vector<std::shared_ptr<SIPLProducerPacket>> _packets;
    std::vector<NvSciBufObj> _buf_objs;
    std::vector<NvSciBufObj> _extra_icp_buf_objs;
    bool _is_ready = false;
    uint32_t _image_ele_index;
    uint32_t _metadata_ele_index;
};

}
}
}