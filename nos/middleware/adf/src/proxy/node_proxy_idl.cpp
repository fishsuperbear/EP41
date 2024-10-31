#include "adf/include/proxy/node_proxy_idl.h"
#include "adf/include/base.h"
#include "adf/include/internal_log.h"

#include "idl/generated/zerocopy_image.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace adf {

#define PROXY_INIT(proxy_ptr, pub_sub_type, domain, topic, receive)                                \
    {                                                                                              \
        proxy_ptr = std::make_shared<hozon::netaos::cm::Proxy>(pub_sub_type);                      \
        if (0 == proxy_ptr->Init(domain, topic)) {                                                 \
            proxy_ptr->Listen(std::bind(&receive, this));                                          \
        } else {                                                                                   \
            ADF_LOG_ERROR << "Init damain ( " << domain << " ), topic ( " << topic << " ) fail !"; \
        }                                                                                          \
    }

#define PROXY_DEINIT(proxy_ptr) \
    { proxy_ptr->Deinit(); }

NodeProxyIdl::NodeProxyIdl(const NodeConfig::CommInstanceConfig& config)
    : NodeProxyBase(config), _domain(config.domain) {
    _topic = config.topic;
    // TODO(zax): not common API for any type.
    if (_topic == "/soc/zerocopy/camera_0" || _topic == "/soc/zerocopy/camera_1") {
        _pub_sub_type = std::make_shared<ZeroCopyImg8M420PubSubType>();
    } else {
        _pub_sub_type = std::make_shared<ZeroCopyImg2M422PubSubType>();
    }

    PROXY_INIT(_proxy, _pub_sub_type, _domain, _topic, NodeProxyIdl::OnDataReceive);
    _freq_monitor.Start();
}

NodeProxyIdl::~NodeProxyIdl() {
    _freq_monitor.Stop();
}

void NodeProxyIdl::OnDataReceive(void) {
    ADF_LOG_INFO << "NodeProxyIdl::OnDataReceive(void) " << _topic;
    std::shared_ptr<BaseData> nvs_image_cuda(new BaseData());

    if (_topic == "/soc/zerocopy/camera_0" || _topic == "/soc/zerocopy/camera_1") {
        int32_t ret = _proxy->Take<ZeroCopyImg8M420>([&nvs_image_cuda, this](const ZeroCopyImg8M420& idl_data) {
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

            nvs_image_cuda->__header.seq = idl_data.frame_count();
            nvs_image_cuda->proto_msg = proto_img;
        });
        if (ret < 0) {
            ADF_LOG_INFO << "Take ZeroCopyImg8M420 error!";
        }
    } else {
        int32_t ret = _proxy->Take<ZeroCopyImg2M422>([&nvs_image_cuda, this](const ZeroCopyImg2M422& idl_data) {
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

            nvs_image_cuda->__header.seq = idl_data.frame_count();
            nvs_image_cuda->proto_msg = proto_img;
        });
        if (ret < 0) {
            ADF_LOG_INFO << "Take ZeroCopyImg2M422 error!";
        }
    }

    PushOneAndNotify(nvs_image_cuda);
    _freq_monitor.PushOnce();
}

void NodeProxyIdl::PauseReceive() {
    PROXY_DEINIT(_proxy);
}

void NodeProxyIdl::ResumeReceive() {
    PROXY_INIT(_proxy, _pub_sub_type, _domain, _topic, NodeProxyIdl::OnDataReceive);
}

void NodeProxyIdl::Deinit() {
    PROXY_DEINIT(_proxy);
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
