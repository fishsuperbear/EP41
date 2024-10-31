#include "adf-lite/ds/ds_recv/idl_cuda_ds_recv.h"
#include "adf/include/node_proto_register.h"
#include "idl/generated/zerocopy_image.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

IdlCudaDsRecv::IdlCudaDsRecv(const DSConfig::DataSource& config) : DsRecv(config), _initialized(false) {
    if (config.cm_topic == "/soc/zerocopy/camera_0" || config.cm_topic == "/soc/zerocopy/camera_1") {
        _proxy.reset(new hozon::netaos::cm::Proxy(std::make_shared<ZeroCopyImg8M420PubSubType>()));
    } else {
        _proxy.reset(new hozon::netaos::cm::Proxy(std::make_shared<ZeroCopyImg2M422PubSubType>()));
    }
    ResumeReceive();
    _writer.Init(_config.topic);
    if (cuda_memory_init == false) {
        cudaStreamCreate(&cuda_stream_);
        DS_LOG_INFO << "cuda memory init success.";
        cuda_memory_init = true;
    } else {
        DS_LOG_DEBUG << "cuda memory already init.";
    }
}

IdlCudaDsRecv::~IdlCudaDsRecv() {}

void IdlCudaDsRecv::Deinit() {
    PauseReceive();
}

void IdlCudaDsRecv::OnDataReceive8M420(void) {
    std::shared_ptr<NvsImageCUDA> nvs_image_cuda(new NvsImageCUDA());

    int32_t ret = _proxy->Take<ZeroCopyImg8M420>([&nvs_image_cuda, this](const ZeroCopyImg8M420& idl_data) {
        uint8_t* cudaPtr = nullptr;
        uint32_t width = idl_data.width();
        uint32_t height = idl_data.height();
        auto img_ptr = idl_data.data().data();
        auto img_size = idl_data.length();
        // 0 -> nv12, 1 -> yuyv
        auto yuv_type = idl_data.yuv_type();

        cudaMalloc((void**)&cudaPtr, img_size);

        DS_LOG_VERBOSE << "cudaPtr " << cudaPtr << " width " << width << " height " << height;

        cudaMemcpyAsync(cudaPtr, img_ptr, img_size, cudaMemcpyHostToDevice, cuda_stream_);

        cudaStreamSynchronize(cuda_stream_);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            DS_LOG_ERROR << "CUDA error " << cudaGetErrorString(error);
            return;
        }

        nvs_image_cuda->data_time_sec = (double)idl_data.sensor_timestamp() / 1000000000;
        nvs_image_cuda->virt_time_sec = 0;
        nvs_image_cuda->width = width;
        nvs_image_cuda->height = height;
        nvs_image_cuda->format = (yuv_type == 0) ? "NV12" : "YUYV";
        nvs_image_cuda->__header.timestamp_real_us = idl_data.pushlish_timestamp() / 1000;
        nvs_image_cuda->__header.timestamp_virt_us = 0;
        nvs_image_cuda->__header.seq = idl_data.frame_count();

        nvs_image_cuda->size = img_size;
        nvs_image_cuda->step = idl_data.stride();
        nvs_image_cuda->cuda_dev_ptr = cudaPtr;
        nvs_image_cuda->need_user_free = true;
    });
    if (ret < 0) {
        DS_LOG_ERROR << "Fail to take cm data of topic " << _config.topic;
        return;
    }
    int32_t alg_ret = _writer.Write(nvs_image_cuda);
    if (alg_ret < 0) {
        DS_LOG_ERROR << "Fail to write " << _config.topic;
        return;
    }

    DS_LOG_VERBOSE << "Recv proto data from cm " << _config.cm_topic;
}

void IdlCudaDsRecv::OnDataReceive2M422(void) {
    std::shared_ptr<NvsImageCUDA> nvs_image_cuda(new NvsImageCUDA());

    int32_t ret = _proxy->Take<ZeroCopyImg2M422>([&nvs_image_cuda, this](const ZeroCopyImg2M422& idl_data) {
        uint8_t* cudaPtr = nullptr;
        uint32_t width = idl_data.width();
        uint32_t height = idl_data.height();
        auto img_ptr = idl_data.data().data();
        auto img_size = idl_data.length();
        // 0 -> nv12, 1 -> yuyv
        auto yuv_type = idl_data.yuv_type();

        cudaMalloc((void**)&cudaPtr, img_size);

        // DS_LOG_VERBOSE << "cudaPtr " << cudaPtr << " width " << width << " height " << height;

        cudaMemcpyAsync(cudaPtr, img_ptr, img_size, cudaMemcpyHostToDevice, cuda_stream_);
        cudaStreamSynchronize(cuda_stream_);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            DS_LOG_ERROR << "CUDA error " << cudaGetErrorString(error);
            return;
        }

        nvs_image_cuda->data_time_sec = (double)idl_data.sensor_timestamp() / 1000000000;
        nvs_image_cuda->virt_time_sec = 0;
        nvs_image_cuda->width = width;
        nvs_image_cuda->height = height;
        nvs_image_cuda->format = (yuv_type == 0) ? "NV12" : "YUYV";
        nvs_image_cuda->__header.timestamp_real_us = idl_data.pushlish_timestamp() / 1000;
        nvs_image_cuda->__header.timestamp_virt_us = 0;
        nvs_image_cuda->__header.seq = idl_data.frame_count();

        nvs_image_cuda->size = img_size;
        nvs_image_cuda->step = idl_data.stride();
        nvs_image_cuda->cuda_dev_ptr = cudaPtr;
        nvs_image_cuda->need_user_free = true;
    });

    if (ret < 0) {
        DS_LOG_ERROR << "Fail to take cm data of topic " << _config.topic;
        return;
    }

    int32_t alg_ret = _writer.Write(nvs_image_cuda);
    if (alg_ret < 0) {
        DS_LOG_ERROR << "Fail to write " << _config.topic;
        return;
    }

    DS_LOG_VERBOSE << "Recv proto data from cm " << _config.cm_topic;
}

std::shared_ptr<NvsImageCUDA> IdlCudaDsRecv::CvtImage2Cuda(const std::shared_ptr<hozon::soc::Image>& pb_Image) {
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
        DS_LOG_ERROR << "Image type is not NV12/YUYV. " << pb_Image->encoding();
        return nullptr;
    }

    if ((image_size == 0) || (image_data.size() != image_size)) {
        DS_LOG_ERROR << "Image recv data size is error."
                     << " image_size : " << image_size << " pb_image_size : " << image_data.size();
        return nullptr;
    }

    if (cuda_memory_init == false) {
        cudaStreamCreate(&cuda_stream_);
        DS_LOG_INFO << "cuda memory init success.";
        cuda_memory_init = true;
    } else {
        DS_LOG_DEBUG << "cuda memory already init.";
    }

    cudaMalloc((void**)&cudaPtr, image_size);

    DS_LOG_VERBOSE << "cudaPtr " << cudaPtr << " width " << width << " height " << height;

    cudaMemcpyAsync(cudaPtr, image_data.c_str(), image_size, cudaMemcpyHostToDevice, cuda_stream_);

    cudaStreamSynchronize(cuda_stream_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        DS_LOG_ERROR << "CUDA error " << cudaGetErrorString(error);
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

void IdlCudaDsRecv::PauseReceive() {
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

void IdlCudaDsRecv::ResumeReceive() {
    if (!_initialized) {
        bool expected = false;
        bool newValue = true;
        bool result = _initialized.compare_exchange_weak(expected, newValue);
        if (result) {
            DS_LOG_INFO << "ResumeReceive: " << _config.cm_topic;
            if (_config.cm_topic == "/soc/zerocopy/camera_0" || _config.cm_topic == "/soc/zerocopy/camera_1") {
                _proxy->Init(_config.cm_domain_id, _config.cm_topic);
                _proxy->Listen(std::bind(&IdlCudaDsRecv::OnDataReceive8M420, this));
            } else {
                _proxy->Init(_config.cm_domain_id, _config.cm_topic);
                _proxy->Listen(std::bind(&IdlCudaDsRecv::OnDataReceive2M422, this));
            }

        } else {
            DS_LOG_INFO << "ResumeReceive do nothing because _initialized value has been true";
        }
    }
}
}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon
