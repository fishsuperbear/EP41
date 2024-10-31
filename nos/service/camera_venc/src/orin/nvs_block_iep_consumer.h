#pragma once

#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include "sensor/nvs_adapter/nvs_block_common.h"
#include "sensor/nvs_adapter/nvs_block_sipl_producer.h"

namespace hozon {
namespace netaos {
namespace nv { 

struct IEPPacket {
    /* The packet handle use for NvSciStream functions */
    NvSciStreamPacket handle;
    /* NvSci buffer object for the packet's data buffer */
    NvSciBufObj nv_sci_buf;
    NvSciBufObj metadata_buf_obj;
    SIPLImageMetadata* metadata_local_ptr;
};

enum PacketConsumeState {
    kPacketConsumed = 0,
    kPacketConsumeAsync,
};

using PacketCb = std::function<int32_t(IEPPacket* packet, NvSciSyncFence& prefence, NvSciSyncFence& eoffence)>;
using GetBufAttrCb = std::function<int32_t(NvSciBufAttrList& buf_attr)>;
using GetWaiterAttrCb = std::function<int32_t(NvSciSyncAttrList& waiter_attr)>;
using GetSignalerAttrCb = std::function<int32_t(NvSciSyncAttrList& signaler_attr)>;
using SetSignalObjCb = std::function<int32_t(NvSciSyncObj& signal_obj)>;
using SetWaiterObjCb = std::function<int32_t(NvSciSyncObj& waiter_obj)>;

using SetBufAttrCb = std::function<int32_t(int32_t elem_type, NvSciBufAttrList buf_attr)>;
using SetBufObjCb = std::function<int32_t(NvSciBufObj)>;

struct IEPConsumerCbs {
    PacketCb packet_cb;
    GetBufAttrCb get_buf_attr_cb;
    GetWaiterAttrCb get_waiter_attr_cb;
    GetSignalerAttrCb get_signaler_attr_cb;
    SetSignalObjCb set_signal_obj_cb;
    SetWaiterObjCb set_waiter_obj_cb;
    SetBufAttrCb set_buf_attr_cb;
    SetBufObjCb set_buf_obj_cb;
};

class NVSBlockIEPConsumer : public NVSBlockCommon {
public:

    int32_t Create(NvSciStreamBlock pool, const std::string& endpoint_info);
    int32_t PacketConsumed(IEPPacket *packet, NvSciSyncFence prefence, NvSciSyncFence eoffence);
    void SetCbs(IEPConsumerCbs& cbs);

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

    int32_t IEPGetBufAttr(NvSciBufAttrList& buf_attr);
    int32_t IEPGetWaiterAttr(NvSciSyncAttrList& waiter_attr);
    int32_t IEPGetSignalAttr(NvSciSyncAttrList& signal_attr);
    int32_t IEPUsePacket(IEPPacket* packet, NvSciSyncObj& signal_obj, NvSciSyncFence& prefence, NvSciSyncFence& eoffence);
    int32_t IEPSetSignalObj(NvSciSyncObj signal_obj);
    int32_t IEPSetWaiterObj(NvSciSyncObj waiter_obj);
    int32_t IEPSetBufAttr(int32_t elem_type, NvSciBufAttrList buf_attr);
    int32_t IEPSetBufObj(NvSciBufObj buf_obj);

    NvSciSyncAttrList _signal_attr;
    NvSciSyncAttrList _waiter_attr;
    NvSciSyncObj _signal_obj;
    NvSciSyncObj _waiter_obj;
    std::vector<std::shared_ptr<IEPPacket>> _packets;
    uint64_t _data_size;
    uint32_t _image_ele_index;
    uint32_t _metadata_ele_index;

    // IEPPacket* _processing_packet = nullptr;
    std::mutex _cbs_mutex;
    IEPConsumerCbs _cbs;
};

}
}
}