#include "CEncManager.h"
#include <mutex>

namespace hozon {
namespace netaos {
namespace desay {

static CEncManager* instance_ = nullptr;
static std::mutex instance_mutex_;

CEncManager& CEncManager::Instance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = new CEncManager();
    }
    return *instance_;
}

void CEncManager::Destroy() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void CEncManager::NotifyEncodedImage(uint32_t sensor_id, Multicast_EncodedImage& encoded_image) {
    if (sensor_enc_cbs_map_.find(sensor_id) != sensor_enc_cbs_map_.end() &&
        sensor_enc_cbs_map_[sensor_id].encoded_image_cb) {
        sensor_enc_cbs_map_[sensor_id].encoded_image_cb(encoded_image);
    }
}

void CEncManager::SetSensorImageCbMap(std::map<uint32_t, EncConsumerCbs> sensor_enc_cbs_map) {
    sensor_enc_cbs_map_ = sensor_enc_cbs_map;
}

void CEncManager::SetCodecType(uint32_t codec_type) {
    codec_type_ = codec_type;
}

uint32_t CEncManager::GetCodecType() {
    return codec_type_;
}

void CEncManager::SetUhpMode(bool uhp_mode) {
    uhp_mode_ = uhp_mode;
}

bool CEncManager::GetUhpMode() {
    return uhp_mode_;
}

void CEncManager::SetSrcLayout(std::map<uint32_t, uint32_t>& src_layout_map) {
    src_layout_map_ = src_layout_map;
}

std::map<uint32_t, uint32_t> CEncManager::GetSrcLayout() {
    return src_layout_map_;
}

void CEncManager::SetFrameSampling(uint32_t frame_sampling) {
    frame_sampling_ = frame_sampling;
}

uint32_t CEncManager::GetFrameSampling() {
    return frame_sampling_;
}

CEncManager::CEncManager()
: codec_type_(0)
, uhp_mode_(false)
, frame_sampling_(0) {

}

CEncManager::~CEncManager() {

}

}
}
}
