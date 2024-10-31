#ifdef BUILD_FOR_ORIN

#include "adf/include/proxy/node_proxy_camera.h"
#include "adf/include/base.h"
#include "adf/include/data_types/common/types.h"
#include "adf/include/internal_log.h"
#include "idl/generated/cm_protobuf.h"
#include "sensor/camera/include/nv_camera.hpp"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeProxyCamera::NodeProxyCamera(const NodeConfig::CommInstanceConfig& config) : NodeProxyBase(config) {
    ADF_LOG_INFO << " config.domain : " << config.domain;
    camera::NvCamera::GetInstance().Init(config.domain);
    camera::NvCamera::GetInstance().RegisterProcess(std::bind(&NodeProxyCamera::OnDataReceive, this), config.domain);
    _freq_monitor.Start();
}

NodeProxyCamera::~NodeProxyCamera() {
    _freq_monitor.Stop();
}

BaseDataTypePtr NodeProxyCamera::CreateBaseDataFromProto(std::shared_ptr<google::protobuf::Message> msg) {
    BaseDataTypePtr base_ptr(new BaseData);
    base_ptr->proto_msg = msg;
    ParseProtoHeader(msg, base_ptr->__header);

    return base_ptr;
}

void NodeProxyCamera::OnDataReceive(void) {
    std::shared_ptr<hozon::soc::Image> yuv_image(new hozon::soc::Image);
    if (yuv_image == nullptr) {
        ADF_LOG_ERROR << "Unknown protobuf type " << _config.name;
        return;
    }

    camera::NvCamera::GetInstance().GetImageData(*(yuv_image->mutable_data()), _config.domain);

    double time_stamp = camera::NvCamera::GetInstance().GetImageTimeStamp(_config.domain);
    uint32_t frame_id = camera::NvCamera::GetInstance().GetFrameID(_config.domain);
    uint32_t height = camera::NvCamera::GetInstance().GetImageHeight(_config.domain);
    uint32_t width = camera::NvCamera::GetInstance().GetImageWidth(_config.domain);

    yuv_image->mutable_header()->set_frame_id(std::to_string(frame_id));
    yuv_image->mutable_header()->set_publish_stamp(time_stamp);
    yuv_image->set_height(height);
    yuv_image->set_width(width);
    yuv_image->set_encoding("NV12");
    yuv_image->set_measurement_time(0);

    ADF_LOG_TRACE << "Proxy receive " << _config.name << " time : " << time_stamp;

    BaseDataTypePtr alg_data = CreateBaseDataFromProto(yuv_image);
    PushOneAndNotify(alg_data);
    _freq_monitor.PushOnce();
}

void NodeProxyCamera::ParseProtoHeader(std::shared_ptr<google::protobuf::Message> proto_msg, Header& header) {
    const google::protobuf::Reflection* reflection = proto_msg->GetReflection();
    const google::protobuf::Descriptor* desc = proto_msg->GetDescriptor();
    const google::protobuf::FieldDescriptor* header_desc = desc->FindFieldByName("header");
    if (!header_desc) {
        ADF_LOG_ERROR << "Missing header in " << proto_msg->GetTypeName();
        return;
    }
    const google::protobuf::Message& header_msg = reflection->GetMessage(*proto_msg, header_desc);
    header.seq = header_msg.GetReflection()->GetUInt32(
        header_msg, ::hozon::common::Header::GetDescriptor()->FindFieldByName("sequence_num"));
    double timestamp_sec = header_msg.GetReflection()->GetDouble(
        header_msg, ::hozon::common::Header::GetDescriptor()->FindFieldByName("timestamp_sec"));
    header.timestamp_real_us = TimestampToUs(timestamp_sec);
}

void NodeProxyCamera::PauseReceive() {}

void NodeProxyCamera::ResumeReceive() {}

void NodeProxyCamera::Deinit() {
    ADF_LOG_DEBUG << "Camera Proxy deinit.";
    camera::NvCamera::GetInstance().DeInit();
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
#endif