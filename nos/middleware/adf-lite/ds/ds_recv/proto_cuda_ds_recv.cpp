#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "adf/include/node_proto_register.h"
#include "adf-lite/ds/ds_recv/proto_cuda_ds_recv.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
    
ProtoCudaDsRecv::ProtoCudaDsRecv(const DSConfig::DataSource& config) :
        DsRecv(config),
        _initialized(false),
        _proxy(std::make_shared<CmProtoBufPubSubType>()) {
    ResumeReceive();
    _writer.Init(_config.topic);
}

ProtoCudaDsRecv::~ProtoCudaDsRecv() {
}

void ProtoCudaDsRecv::Deinit() {
    PauseReceive();
}

void ProtoCudaDsRecv::OnDataReceive(void) {
    std::shared_ptr<CmProtoBuf> cm_data(new CmProtoBuf);
    if (cm_data == nullptr) {
        DS_LOG_ERROR << "cm_data pointer is nullptr!!!";
    }
    int32_t ret = _proxy.Take(cm_data);
    if (ret < 0) {
        DS_LOG_ERROR << "Fail to take cm data of topic " << _config.topic;
        return;
    }

    std::shared_ptr<google::protobuf::Message> msg = adf::ProtoMessageTypeMgr::GetInstance().Create(_config.topic);
    if (msg == nullptr) {
        DS_LOG_ERROR << "Fail to find topic prototype " << _config.topic;
        return;
    }

    bool bret = msg->ParseFromArray(cm_data->str().data(), cm_data->str().size());
    if (bret == false) {
        DS_LOG_ERROR << "Fail to parse proto " << _config.topic;
        return;
    }

    BaseDataTypePtr base_ptr = CvtImage2Cuda(std::static_pointer_cast<hozon::soc::Image>(msg));
    if (base_ptr == nullptr) {
        DS_LOG_ERROR << "base_ptr pointer is nullptr!!!";
    }

    DS_LOG_DEBUG << "check header info: data->__header.timestamp_real_us = " << base_ptr->__header.timestamp_real_us;

    // 将cm_data中的header信息复制到base_ptr中
    if (cm_data->header().latency_info().link_infos().size() > 0) {

        for (auto linkinfo : cm_data->header().latency_info().link_infos()) {
            DS_LOG_DEBUG << "base_ptr->__header.latency_info().link_infos():"
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].sec << " "
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].nsec;
            base_ptr->__header.latency_info.data[linkinfo.link_name()].sec = linkinfo.timestamp_real().sec();
            base_ptr->__header.latency_info.data[linkinfo.link_name()].nsec = linkinfo.timestamp_real().nsec();
            DS_LOG_DEBUG << "base_ptr->__header.latency_info().link_infos():"
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].sec << " "
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].nsec;
        }
    } else {
        DS_LOG_DEBUG << "_config.topic:" << _config.topic << " cm_data->header has not link_infos, size is: "
                     << cm_data->header().latency_info().link_infos().size();
    }

    int32_t alg_ret = _writer.Write(base_ptr);
    if (alg_ret < 0) {
        DS_LOG_ERROR << "Fail to write " << _config.topic;
        return ;
    } else {
        if (base_ptr != nullptr) {
            DS_LOG_VERBOSE << "Success write data " << _config.topic << " timestamp is " << base_ptr->__header.timestamp_real_us;
        }
    }

    DS_LOG_VERBOSE << "Recv proto data from cm " << _config.cm_topic;
}

std::shared_ptr<NvsImageCUDA> ProtoCudaDsRecv::CvtImage2Cuda(const std::shared_ptr<hozon::soc::Image> &pb_Image) {
    std::shared_ptr<NvsImageCUDA> _nvs_image_cuda(new NvsImageCUDA());

    uint8_t *cudaPtr = nullptr;
    uint32_t image_size = 0;
    uint32_t width = pb_Image->width();
    uint32_t height = pb_Image->height();
    const std::string &image_data = pb_Image->data();

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
                    << " image_size : " << image_size
                    << " pb_image_size : " << image_data.size();
        return nullptr;
    }

    if (cuda_memory_init == false) {
        cudaStreamCreate(&cuda_stream_);
        DS_LOG_INFO << "cuda memory init success.";
        cuda_memory_init = true;
    } else {
        DS_LOG_DEBUG << "cuda memory already init.";
    }

    cudaMalloc((void **)&cudaPtr, image_size);

    DS_LOG_VERBOSE << "cudaPtr " << cudaPtr << " width " << width
                << " height " << height;

    cudaMemcpyAsync(cudaPtr, image_data.c_str(), image_size,
            cudaMemcpyHostToDevice, cuda_stream_);

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

void ProtoCudaDsRecv::PauseReceive() {
    if (_initialized) {
        bool expected = true;
        bool newValue = false;
        bool result = _initialized.compare_exchange_weak(expected, newValue);
        if (result) {
            DS_LOG_INFO << "PauseReceive: " << _config.cm_topic;
            _proxy.Deinit();
        }
    } else {
        DS_LOG_INFO << "PauseReceive do nothing because _initialized value has been false";
    }
}

void ProtoCudaDsRecv::ResumeReceive() {
    if (!_initialized) {
        bool expected = false;
        bool newValue = true;
        bool result = _initialized.compare_exchange_weak(expected, newValue);
        if (result) {
            DS_LOG_INFO << "ResumeReceive: " << _config.cm_topic;
            _proxy.Init(_config.cm_domain_id, _config.cm_topic);
            _proxy.Listen(std::bind(&ProtoCudaDsRecv::OnDataReceive, this));
        } else {
            DS_LOG_INFO << "ResumeReceive do nothing because _initialized value has been true";
        }
    }
}
}
}
}

