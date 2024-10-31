#include "adf-lite/ds/ds_recv/idl_ds_recv.h"
#include "adf/include/node_proto_register.h"
#include "idl/generated/zerocopy_image.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

IdlDsRecv::IdlDsRecv(const DSConfig::DataSource& config) : DsRecv(config), _initialized(false) {
    if (config.cm_topic == "/soc/zerocopy/camera_0" || config.cm_topic == "/soc/zerocopy/camera_1") {
        _proxy.reset(new hozon::netaos::cm::Proxy(std::make_shared<ZeroCopyImg8M420PubSubType>()));
    } else {
        _proxy.reset(new hozon::netaos::cm::Proxy(std::make_shared<ZeroCopyImg2M422PubSubType>()));
    }
    ResumeReceive();
    _writer.Init(_config.topic);
}

IdlDsRecv::~IdlDsRecv() {}

void IdlDsRecv::Deinit() {
    PauseReceive();
}

void IdlDsRecv::OnDataReceive(void) {
    DS_LOG_INFO << "IdlDsRecv::OnDataReceive(void) " << _config.topic;
    std::shared_ptr<BaseData> idl_image(new BaseData());

    if (_config.topic == "/soc/zerocopy/camera_0" || _config.topic == "/soc/zerocopy/camera_1") {
        int32_t ret = _proxy->Take<ZeroCopyImg8M420>([&idl_image, this](const ZeroCopyImg8M420& idl_data) {
            auto proto_img = std::make_shared<hozon::soc::Image>();
            auto header = proto_img->mutable_header();
            header->mutable_sensor_stamp()->set_camera_stamp(idl_data.sensor_timestamp() / 1000000000.0);
            header->set_publish_stamp(idl_data.pushlish_timestamp() / 1000000000.0);
            auto img_data = proto_img->mutable_data();
            img_data->assign(idl_data.data().data(), idl_data.length());
            header->set_seq(idl_data.frame_count());
            proto_img->set_encoding("NV12");
            proto_img->set_height(idl_data.height());
            proto_img->set_width(idl_data.width());

            idl_image->__header.seq = idl_data.frame_count();
            idl_image->proto_msg = proto_img;
        });
        if (ret < 0) {
            DS_LOG_INFO << "Take ZeroCopyImg8M420 error!";
        }
    } else {
        int32_t ret = _proxy->Take<ZeroCopyImg2M422>([&idl_image, this](const ZeroCopyImg2M422& idl_data) {
            auto proto_img = std::make_shared<hozon::soc::Image>();
            auto header = proto_img->mutable_header();
            header->mutable_sensor_stamp()->set_camera_stamp(idl_data.sensor_timestamp() / 1000000000.0);
            header->set_publish_stamp(idl_data.pushlish_timestamp() / 1000000000.0);
            auto img_data = proto_img->mutable_data();
            img_data->assign(idl_data.data().data(), idl_data.length());
            header->set_seq(idl_data.frame_count());
            proto_img->set_encoding("YUYV");
            proto_img->set_height(idl_data.height());
            proto_img->set_width(idl_data.width());

            idl_image->__header.seq = idl_data.frame_count();
            idl_image->proto_msg = proto_img;
        });
        if (ret < 0) {
            DS_LOG_INFO << "Take ZeroCopyImg2M422 error!";
        }
    }
    int32_t alg_ret = _writer.Write(idl_image);
    if (alg_ret < 0) {
        DS_LOG_ERROR << "Fail to write " << _config.topic;
        return;
    }
}

void IdlDsRecv::PauseReceive() {
    if (_initialized) {
        bool expected = true;
        bool newValue = false;
        bool result = _initialized.compare_exchange_weak(expected, newValue);
        if (result) {
            DS_LOG_INFO << "PauseReceive: " << _config.cm_topic;
            _proxy->Deinit();
        }
    } else {
        DS_LOG_INFO << "PauseReceive do nothing because _initialized value has been false";
    }
}

void IdlDsRecv::ResumeReceive() {
    if (!_initialized) {
        bool expected = false;
        bool newValue = true;
        bool result = _initialized.compare_exchange_weak(expected, newValue);
        if (result) {
            DS_LOG_INFO << "ResumeReceive: " << _config.cm_topic;
            _proxy->Init(_config.cm_domain_id, _config.cm_topic);
            _proxy->Listen(std::bind(&IdlDsRecv::OnDataReceive, this));

        } else {
            DS_LOG_INFO << "ResumeReceive do nothing because _initialized value has been true";
        }
    }
}
}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon
