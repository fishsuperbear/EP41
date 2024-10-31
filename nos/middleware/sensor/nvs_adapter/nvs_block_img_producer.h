#pragma once

#include "sensor/nvs_adapter/nvs_block_common.h"
#include <vector>

namespace hozon {
namespace netaos {
namespace nv { 

struct ImageProducerPacket {
    NvSciStreamPacket handle;
    NvSciBufObj nv_sci_buf;
};

class NVSBlockImgProducer : public NVSBlockCommon {
public:
    NVSBlockImgProducer();
    ~NVSBlockImgProducer();

    int32_t Create(NvSciStreamBlock pool, const std::string& endpoint_info);

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
    int32_t StreamInit();
    int32_t ProducerInit();
    int32_t ProducerElemSupport();
    int32_t ProducerSyncSupport();

    uint32_t _num_consumers;
    NvSciBufAttrList _source_attr;
    NvSciSyncAttrList _signal_attr;
    NvSciSyncAttrList _waiter_attr;
    NvSciSyncObj _signal_obj;
    NvSciSyncObj _waiter_obj[MAX_CONSUMERS];
    NvSciSyncCpuWaitContext _cpu_wait_context;
    uint32_t _num_packet = 0;
    std::vector<std::shared_ptr<ImageProducerPacket>> _packets;
    uint32_t _counter = 0;
};

}
}
}