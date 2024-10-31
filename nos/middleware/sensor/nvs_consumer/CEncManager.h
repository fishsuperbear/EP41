#ifndef CENCMANAGER_H
#define CENCMANAGER_H

#include <map>
#include <functional>
#include <string>

namespace hozon {
namespace netaos {
namespace desay {

struct Multicast_EncodedImage {
    uint32_t seq;
    uint64_t sensor_time;
    double recv_time;
    uint32_t frame_type;
    std::string frame_id;
    std::string data;
};
using EncodedImageCb = std::function<void(Multicast_EncodedImage& encoded_image)>;

struct EncConsumerCbs {
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

enum CodecType {
    kCodecH264 = 0,
    kCodecH265
};

class CEncManager
{
public:
    static CEncManager& Instance();
    static void Destroy();

    void NotifyEncodedImage(uint32_t sensor_id, Multicast_EncodedImage& encoded_image);
    void SetSensorImageCbMap(std::map<uint32_t, EncConsumerCbs> sensor_enc_cbs_map);
    void SetCodecType(uint32_t codec_type);
    uint32_t GetCodecType();
    void SetUhpMode(bool uhp_mode);
    bool GetUhpMode();
    void SetSrcLayout(std::map<uint32_t, uint32_t>& src_layout_map);
    std::map<uint32_t, uint32_t> GetSrcLayout();
    void SetFrameSampling(uint32_t frame_sampling);
    uint32_t GetFrameSampling();
private:
    CEncManager();
    ~CEncManager();

    uint32_t codec_type_;
    bool uhp_mode_;
    std::map<uint32_t, EncConsumerCbs> sensor_enc_cbs_map_;
    std::map<uint32_t, uint32_t> src_layout_map_;
    uint32_t frame_sampling_;
};

}
}
}

#endif
