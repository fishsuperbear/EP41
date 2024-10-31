#include "adf/include/proxy/node_proxy_h265_to_yuv.h"
#include "adf/include/base.h"
#include "adf/include/data_types/common/types.h"
#include "adf/include/internal_log.h"
#include "idl/generated/cm_protobuf.h"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeProxyH265ToYUV::NodeProxyH265ToYUV(const NodeConfig::CommInstanceConfig& config,
                                       std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type)
    : NodeProxyProto(config, pub_sub_type) {
    _decoder = codec::DecoderFactory::Create();
    if (!_decoder) {
        ADF_LOG_CRITICAL << "Fail to create decoder, " << config.name;
    }

    int32_t ret = _decoder->Init("");
    if (ret < 0) {
        ADF_LOG_CRITICAL << "Fail to init decoder " << ret << ", " << config.name;
    }

    _freq_monitor.Start();
}

NodeProxyH265ToYUV::~NodeProxyH265ToYUV() {
    _freq_monitor.Stop();
}

void NodeProxyH265ToYUV::OnDataReceive(void) {
    std::shared_ptr<hozon::soc::CompressedImage> h265_image(new hozon::soc::CompressedImage);
    std::shared_ptr<hozon::soc::Image> yuv_image(new hozon::soc::Image);
    std::shared_ptr<CmProtoBuf> idl_msg(new CmProtoBuf);
    _proxy->Take(idl_msg);
    ADF_LOG_TRACE << "Proxy receive " << _config.name;

    bool ret = h265_image->ParseFromArray(idl_msg->str().data(), idl_msg->str().size());
    if (!ret) {
        ADF_LOG_ERROR << "Fail to parse protobuf " << _config.name;
        return;
    }

    // std::chrono::steady_clock::time_point begin_tp = std::chrono::steady_clock::now();
    int32_t iret = _decoder->Process(h265_image->data(), *(yuv_image->mutable_data()));
    if (iret < 0) {
        ADF_LOG_ERROR << "Fail to decode " << iret << ", " << _config.name;
        return;
    }
    // std::chrono::steady_clock::time_point end_tp = std::chrono::steady_clock::now();
    // ADF_LOG_INFO << "Decode time cost " << std::chrono::duration<double, std::milli>(end_tp - begin_tp).count() << "(ms).";
    ADF_LOG_TRACE << "Decode succ " << _config.name << ", h265 size: " << h265_image->data().size()
                  << ", input size: " << h265_image->data().size() << ", output size: " << yuv_image->data().size()
                  << ", width: " << _decoder->GetWidth() << ", height: " << _decoder->GetHeight();

    yuv_image->mutable_header()->set_publish_stamp(h265_image->mutable_header()->publish_stamp());
    yuv_image->set_height(_decoder->GetHeight());
    yuv_image->set_width(_decoder->GetWidth());
    yuv_image->set_encoding("NV12");
    yuv_image->set_measurement_time(0.1);

    BaseDataTypePtr base_ptr = CreateBaseDataFromProto(yuv_image);
    PushOneAndNotify(base_ptr);
    _freq_monitor.PushOnce();
    // checker.say(_config.name);

    // std::stringstream ss;
    // ss << std::fixed << yuv_image->mutable_header()->timestamp_sec() * 1000;
    // ADF_LOG_INFO << _config.name << " ts " << ss.str();
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon