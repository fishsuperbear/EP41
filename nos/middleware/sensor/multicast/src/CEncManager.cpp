#include "CEncManager.h"
#include <mutex>

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

void CEncManager::NotifyEncodedImage(int32_t sensor_id, hozon::netaos::multicast::Multicast_EncodedImage& encoded_image) {
    if (sensor_enc_cbs_map_.find(sensor_id) != sensor_enc_cbs_map_.end() &&
        sensor_enc_cbs_map_[sensor_id].encoded_image_cb) {
        sensor_enc_cbs_map_[sensor_id].encoded_image_cb(encoded_image);
    }
}

void CEncManager::SetSensorImageCbMap(std::map<int, hozon::netaos::multicast::IEPConsumerCbs> sensor_enc_cbs_map) {
    sensor_enc_cbs_map_ = sensor_enc_cbs_map;
}

CEncManager::CEncManager() {

}

CEncManager::~CEncManager() {

}
