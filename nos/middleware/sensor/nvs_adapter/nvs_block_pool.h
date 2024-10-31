#pragma once

#include "sensor/nvs_adapter/nvs_block_common.h"

namespace hozon {
namespace netaos {
namespace nv { 

class NVSBlockPool : public NVSBlockCommon {
public:
    int32_t Create(uint32_t num_packet);

protected:
    virtual int32_t OnConnected() override;
    virtual int32_t OnElements() override;
    virtual int32_t OnPacketStatus() override;

private:
    uint32_t _num_packet;
    uint32_t _num_consumers;
    uint32_t _num_producer_elem;
    uint32_t _num_consumer_elem;
    ElemAttr _producer_elem[MAX_ELEMS];
    ElemAttr _consumer_elem[MAX_ELEMS];
    NvSciStreamPacket _packet[MAX_PACKETS];
    uint32_t _num_packet_ready = 0;
};

}
}
}