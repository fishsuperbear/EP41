#include "adf/include/proxy/node_proxy_proto_cuda.h"
#include "adf/include/base.h"
#include "adf/include/data_types/common/types.h"
#include "adf/include/internal_log.h"
#include "idl/generated/cm_protobuf.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeProxyProtoCuda::NodeProxyProtoCuda(const NodeConfig::CommInstanceConfig& config,
                                       std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type)
    : NodeProxyCM(config, pub_sub_type) {
    _freq_monitor.Start();
}

NodeProxyProtoCuda::~NodeProxyProtoCuda() {
    _freq_monitor.Stop();
}

void NodeProxyProtoCuda::OnDataReceive(void) {
    std::shared_ptr<google::protobuf::Message> proto_msg = ProtoMessageTypeMgr::GetInstance().Create(_config.name);
    if (proto_msg == nullptr) {
        ADF_LOG_ERROR << "Unknown protobuf type " << _config.name;
        return;
    }

    std::shared_ptr<CmProtoBuf> idl_msg(new CmProtoBuf);
    int32_t ret = _proxy->Take(idl_msg);
    if (ret < 0) {
        ADF_LOG_ERROR << "Fail to take cm data of topic " << _config.topic;
        return;
    }

    ADF_LOG_TRACE << "Proxy receive " << _config.name;

    bool msg_ret = proto_msg->ParseFromArray(idl_msg->str().data(), idl_msg->str().size());
    if (msg_ret == false) {
        ADF_LOG_ERROR << "Fail to parse protobuf " << _config.name;
        return;
    }

    BaseDataTypePtr alg_data = CvtImage2Cuda(std::static_pointer_cast<hozon::soc::Image>(proto_msg));

    PushOneAndNotify(alg_data);
    _freq_monitor.PushOnce();
}

std::shared_ptr<NvsImageCUDA> NodeProxyProtoCuda::CvtImage2Cuda(const std::shared_ptr<hozon::soc::Image>& pb_Image) {
    std::shared_ptr<NvsImageCUDA> _nvs_image_cuda(new NvsImageCUDA());

    uint8_t* cudaPtr = nullptr;
    uint32_t image_size = 0;
    uint32_t width = pb_Image->width();
    uint32_t height = pb_Image->height();
    const std::string& image_data = pb_Image->data();

    if (pb_Image->encoding() == "NV12") {
        image_size = width * height * 3 / 2;
    } else if (pb_Image->encoding() == "YUYV") {
        image_size = width * height * 2;
    } else {
        ADF_LOG_ERROR << "Image type is not NV12/YUYV. " << pb_Image->encoding();
        return nullptr;
    }

    if ((image_size == 0) || (image_data.size() != image_size)) {
        ADF_LOG_ERROR << "Image recv data size is error."
                      << " image_size : " << image_size << " pb_image_size : " << image_data.size();
        return nullptr;
    }

    if (cuda_memory_init == false) {
        cudaStreamCreate(&cuda_stream_);
        ADF_LOG_INFO << "cuda memory init success.";
        cuda_memory_init = true;
    } else {
        ADF_LOG_DEBUG << "cuda memory already init.";
    }

    cudaMalloc((void**)&cudaPtr, image_size);

    ADF_LOG_TRACE << "cudaPtr " << cudaPtr << " width " << width << " height " << height;

    cudaMemcpyAsync(cudaPtr, image_data.c_str(), image_size, cudaMemcpyHostToDevice, cuda_stream_);

    cudaStreamSynchronize(cuda_stream_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        ADF_LOG_ERROR << "CUDA error " << cudaGetErrorString(error);
        return nullptr;
    }

    _nvs_image_cuda->data_time_sec = pb_Image->header().sensor_stamp().camera_stamp();
    _nvs_image_cuda->virt_time_sec = 0;
    _nvs_image_cuda->width = pb_Image->width();
    _nvs_image_cuda->height = pb_Image->height();
    _nvs_image_cuda->format = pb_Image->encoding();
    _nvs_image_cuda->__header.timestamp_real_us = pb_Image->header().publish_stamp() * 1000 * 1000;
    _nvs_image_cuda->__header.timestamp_virt_us = 0;
    _nvs_image_cuda->__header.seq = pb_Image->header().seq();

    _nvs_image_cuda->size = image_size;
    _nvs_image_cuda->step = pb_Image->step();
    _nvs_image_cuda->cuda_dev_ptr = cudaPtr;
    _nvs_image_cuda->need_user_free = true;

    return _nvs_image_cuda;
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon