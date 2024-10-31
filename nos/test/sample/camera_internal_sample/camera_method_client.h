#include <cstdint>
#include <memory>
#include "cm/include/method.h"
#include "idl/generated/camera_internal_dataPubSubTypes.h"

#define SENSOR_0X8B40 "OX08B40"
#define SENSOR_ISX031 "isx031"
#define SENSOR_ISX021 "isx021"

struct CameraInternalData {
    bool isValid;
    std::uint8_t sensor_id;
    std::string module_name;
    std::vector<uint8_t> data;
};

class CameraMethodClient {
public:
    CameraMethodClient() :
        _camera_client(std::make_shared<camera_internal_data_requestPubSubType>(), std::make_shared<camera_internal_data_replyPubSubType>()) {
        _camera_client.Init(2, "camera_interal_data");
    }

    int32_t DeInit() {
        return _camera_client.Deinit();
    }

    int32_t WaitServiceOnline(int64_t timeout_ms) {
        return _camera_client.WaitServiceOnline(timeout_ms);
    }

    int32_t GetCameraInternalData(uint8_t sensor_id, CameraInternalData& data) {
        int64_t timeout_ms = 1000;
        std::shared_ptr<camera_internal_data_request> camera_request = std::make_shared<camera_internal_data_request>();
        std::shared_ptr<camera_internal_data_reply> camera_replay = std::make_shared<camera_internal_data_reply>();
        camera_request->sensor_id(sensor_id);

        int ret = _camera_client.Request(camera_request, camera_replay, timeout_ms);
        if (ret < 0) {
            return -1;
        }

        data.isValid = camera_replay->isvalid();
        data.module_name = camera_replay->module_name();
        data.sensor_id = camera_replay->sensor_id();
        data.data = camera_replay->data();

        return 0;
    }

private:
    hozon::netaos::cm::Client<camera_internal_data_request, camera_internal_data_reply> _camera_client;
};