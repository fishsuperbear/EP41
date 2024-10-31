#include <functional>
#include <map>
#include <vector>
#include "nvscibuf.h"

#pragma once

namespace hozon {
namespace netaos {
namespace multicast {

int multicast_init(int argc, char *argv[]);
int multicast_quit();

// for encoder consumer in multicast
// using PacketCb = std::function<int32_t(IEPPacket* packet, NvSciSyncFence& prefence, NvSciSyncFence& eoffence)>;
// using GetBufAttrCb = std::function<int32_t(NvSciBufAttrList& buf_attr)>;
// using GetWaiterAttrCb = std::function<int32_t(NvSciSyncAttrList& waiter_attr)>;
// using GetSignalerAttrCb = std::function<int32_t(NvSciSyncAttrList& signaler_attr)>;
// using SetSignalObjCb = std::function<int32_t(NvSciSyncObj& signal_obj)>;
// using SetWaiterObjCb = std::function<int32_t(NvSciSyncObj& waiter_obj)>;

// using SetBufAttrCb = std::function<int32_t(int32_t elem_type, NvSciBufAttrList buf_attr)>;
// using SetBufObjCb = std::function<int32_t(NvSciBufObj)>;

struct Multicast_EncodedImage {
    std::string data;
};
using EncodedImageCb = std::function<void(Multicast_EncodedImage& encoded_image)>;

struct IEPConsumerCbs {
    // PacketCb packet_cb;
    // GetBufAttrCb get_buf_attr_cb;
    // GetWaiterAttrCb get_waiter_attr_cb;
    // GetSignalerAttrCb get_signaler_attr_cb;
    // SetSignalObjCb set_signal_obj_cb;
    // SetWaiterObjCb set_waiter_obj_cb;
    // SetBufAttrCb set_buf_attr_cb;
    // SetBufObjCb set_buf_obj_cb;
    EncodedImageCb encoded_image_cb;
};

// Must set before init and never set again due to the function is not thread-safe.
int multicast_set_enc_cbs(std::map<int, IEPConsumerCbs> sensor_enc_cbs_map);

}
}
}