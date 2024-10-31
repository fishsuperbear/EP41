#include "adf/include/proxy/node_proxy_idl_cuda.h"
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

NodeProxyIdlCuda::NodeProxyIdlCuda(const NodeConfig::CommInstanceConfig& config)
    : NodeProxyBase(config), _domain(config.domain) {
    _topic = config.topic;
    // TODO(zax): not common API for any type.
    if (_topic == "/soc/zerocopy/camera_0" || _topic == "/soc/zerocopy/camera_1") {
        _pub_sub_type = std::make_shared<ZeroCopyImg8M420PubSubType>();
    } else {
        _pub_sub_type = std::make_shared<ZeroCopyImg2M422PubSubType>();
    }

    PROXY_INIT(_proxy, _pub_sub_type, _domain, _topic, NodeProxyIdlCuda::OnDataReceive);
    _freq_monitor.Start();

    if (_cuda_memory_init == false) {
        cudaStreamCreate(&_cuda_stream);
        ADF_LOG_INFO << "cuda memory init success.";
        _cuda_memory_init = true;
    } else {
        ADF_LOG_INFO << "cuda memory already init.";
    }
}

NodeProxyIdlCuda::~NodeProxyIdlCuda() {
    _freq_monitor.Stop();
}

void NodeProxyIdlCuda::OnDataReceive(void) {
    ADF_LOG_INFO << "NodeProxyIdlCuda::OnDataReceive(void) " << _topic;
    std::shared_ptr<NvsImageCUDA> nvs_image_cuda(new NvsImageCUDA());
    if (_topic == "/soc/zerocopy/camera_0" || _topic == "/soc/zerocopy/camera_1") {
        int32_t ret = _proxy->Take<ZeroCopyImg8M420>([&nvs_image_cuda, this](const ZeroCopyImg8M420& idl_data) {
            // set cuda
            uint8_t* cudaPtr = nullptr;
            auto img_size = idl_data.length();
            cudaMalloc((void**)&cudaPtr, img_size);
            cudaMemcpyAsync(cudaPtr, idl_data.data().data(), img_size, cudaMemcpyHostToDevice, _cuda_stream);
            cudaStreamSynchronize(_cuda_stream);
            auto error = cudaGetLastError();
            if (error != cudaSuccess) {
                ADF_LOG_INFO << "CUDA error " << cudaGetErrorString(error);
                return;
            }
            nvs_image_cuda->data_time_sec = (double)idl_data.sensor_timestamp() / 1000000000;
            nvs_image_cuda->virt_time_sec = 0;
            nvs_image_cuda->width = idl_data.width();
            nvs_image_cuda->height = idl_data.height();
            nvs_image_cuda->format = (idl_data.yuv_type() == 0) ? "NV12" : "YUYV";
            nvs_image_cuda->__header.timestamp_real_us = idl_data.pushlish_timestamp() / 1000;
            nvs_image_cuda->__header.timestamp_virt_us = 0;
            nvs_image_cuda->__header.seq = idl_data.frame_count();

            nvs_image_cuda->size = img_size;
            nvs_image_cuda->step = idl_data.stride();
            nvs_image_cuda->cuda_dev_ptr = cudaPtr;
            nvs_image_cuda->need_user_free = true;
        });
        if (ret < 0) {
            ADF_LOG_INFO << "Take ZeroCopyImg8M420 error!";
        }
    } else {
        int32_t ret = _proxy->Take<ZeroCopyImg2M422>([&nvs_image_cuda, this](const ZeroCopyImg2M422& idl_data) {
            // set cuda
            uint8_t* cudaPtr = nullptr;
            auto img_size = idl_data.length();
            cudaMalloc((void**)&cudaPtr, img_size);
            cudaMemcpyAsync(cudaPtr, idl_data.data().data(), img_size, cudaMemcpyHostToDevice, _cuda_stream);
            cudaStreamSynchronize(_cuda_stream);
            auto error = cudaGetLastError();
            if (error != cudaSuccess) {
                ADF_LOG_INFO << "CUDA error " << cudaGetErrorString(error);
                return;
            }
            nvs_image_cuda->data_time_sec = (double)idl_data.sensor_timestamp() / 1000000000;
            nvs_image_cuda->virt_time_sec = 0;
            nvs_image_cuda->width = idl_data.width();
            nvs_image_cuda->height = idl_data.height();
            nvs_image_cuda->format = (idl_data.yuv_type() == 0) ? "NV12" : "YUYV";
            nvs_image_cuda->__header.timestamp_real_us = idl_data.pushlish_timestamp() / 1000;
            nvs_image_cuda->__header.timestamp_virt_us = 0;
            nvs_image_cuda->__header.seq = idl_data.frame_count();

            nvs_image_cuda->size = img_size;
            nvs_image_cuda->step = idl_data.stride();
            nvs_image_cuda->cuda_dev_ptr = cudaPtr;
            nvs_image_cuda->need_user_free = true;
        });
        if (ret < 0) {
            ADF_LOG_INFO << "Take ZeroCopyImg2M422 error!";
        }
    }

    PushOneAndNotify(nvs_image_cuda);
    _freq_monitor.PushOnce();
}

void NodeProxyIdlCuda::PauseReceive() {
    PROXY_DEINIT(_proxy);
}

void NodeProxyIdlCuda::ResumeReceive() {
    PROXY_INIT(_proxy, _pub_sub_type, _domain, _topic, NodeProxyIdlCuda::OnDataReceive);
}

void NodeProxyIdlCuda::Deinit() {
    PROXY_DEINIT(_proxy);
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
