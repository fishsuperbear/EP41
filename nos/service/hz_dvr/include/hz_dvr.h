#pragma once

#ifdef BUILD_FOR_ORIN

#include <yaml-cpp/yaml.h>
#include "hz_dvr_logger.h"
#include "sensor/nvs_adapter/nvs_helper.h"
#include "sensor/nvs_consumer/CIpcConsumerChannel.hpp"
#include "idl/generated/sensor_reattachPubSubTypes.h"
#include "cm/include/method.h"

namespace hozon {
namespace netaos {
namespace hz_dvr {

class Dvr {
   public:
    static Dvr& GetInstance() {
        static Dvr instance;
        return instance;
    }

    ~Dvr() = default;

    int Init();
    void Run();
    void Deinit();

    void SetParameter(const uint32_t& width, const uint32_t& height, const uint32_t& sensor_id, const uint32_t& channel_id) {
        width_ = width;
        height_ = height;
        sensor_info_.id = sensor_id;
        channel_id_ = channel_id;
    };

    uint32_t GetSensorId() {
        return sensor_info_.id;
    }

    uint32_t GetChannelId() {
        return channel_id_;
    }

    void GetDisplayConfig(uint32_t& width, uint32_t& height) {
        width = width_;
        height = height_;
    }

   private:
    Dvr() = default;
    std::shared_ptr<desay::CIpcConsumerChannel> consumer_;
    SensorInfo sensor_info_;
    uint16_t width_ = 1920;
    uint16_t height_ = 1536;
    uint32_t channel_id_ = 4;
    std::unique_ptr<hozon::netaos::cm::Client<sensor_reattach, sensor_reattach_resp>> attach_client_ptr_;
};
}  // namespace hz_dvr
}  // namespace netaos
}  // namespace hozon
#endif
