#include "hz_dvr.h"
#include <yaml-cpp/yaml.h>
namespace hozon {
namespace netaos {
namespace hz_dvr {
int Dvr::Init() {
    sensor_info_.id = GetSensorId();
    std::shared_ptr<sensor_reattachPubSubType> req_data_type = std::make_shared<sensor_reattachPubSubType>();
    std::shared_ptr<sensor_reattach_respPubSubType> resp_data_type = std::make_shared<sensor_reattach_respPubSubType>();
    std::shared_ptr<sensor_reattach> req_data = std::make_shared<sensor_reattach>();
    attach_client_ptr_.reset(new hozon::netaos::cm::Client<sensor_reattach, sensor_reattach_resp>(req_data_type, resp_data_type));
    attach_client_ptr_->Init(0, "sensor_reattach");

    attach_client_ptr_->WaitServiceOnline(5000);

    hozon::netaos::nv::NVSHelper::GetInstance().Init();
    consumer_ = std::make_shared<hozon::netaos::desay::CIpcConsumerChannel>(
        hozon::netaos::nv::NVSHelper::GetInstance().sci_buf_module,
        hozon::netaos::nv::NVSHelper::GetInstance().sci_sync_module, &sensor_info_,
        hozon::netaos::desay::DISPLAY_CONSUMER, sensor_info_.id, GetChannelId());

    auto status = consumer_->CreateBlocks(nullptr);

    hozon::netaos::desay::ConsumerConfig consumer_config{false, true};
    consumer_->SetConsumerConfig(consumer_config);

    hozon::netaos::desay::DisplayConfig display_config{0};
    display_config.bzoomFlag = true;
    GetDisplayConfig(display_config.zoom_width, display_config.zoom_height);
    DVR_LOG_INFO << "display width :" << display_config.zoom_width;
    DVR_LOG_INFO << "display height :" << display_config.zoom_height;
    consumer_->SetDisplayConfig(display_config);

    req_data->isattach(true);
    req_data->sensor_id(sensor_info_.id);
    req_data->index(GetChannelId());
    attach_client_ptr_->RequestAndForget(req_data);

    status = consumer_->Connect();
    status = consumer_->InitBlocks();
    status = consumer_->Reconcile();
    return status;
}

void Dvr::Run() {
    consumer_->Start();
}

void Dvr::Deinit() {
    std::shared_ptr<sensor_reattach> req_data = std::make_shared<sensor_reattach>();
    req_data->isattach(false);
    req_data->sensor_id(sensor_info_.id);
    req_data->index(GetChannelId());
    attach_client_ptr_->RequestAndForget(req_data);
    consumer_->Stop();
    consumer_->Deinit();
    attach_client_ptr_->Deinit();
}
}  // namespace hz_dvr
}  // namespace netaos
}  // namespace hozon
