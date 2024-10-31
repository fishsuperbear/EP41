#ifdef BUILD_FOR_ORIN

#include <stdio.h>
#include <mutex>

#include "adf-lite/ds/ds_recv/nvs_cuda_ds_recv.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

NvsCudaDesayDsRecv::NvsCudaDesayDsRecv(const DSConfig::DataSource& config) : DsRecv(config) {
    nv::NVSHelper::GetInstance().Init();
    InitNVS(config.cm_topic);
    _writer.Init(_config.topic);
}

NvsCudaDesayDsRecv::~NvsCudaDesayDsRecv() {}

void NvsCudaDesayDsRecv::OnDataReceive(void) {
    return;
}

void NvsCudaDesayDsRecv::PauseReceive() {
    return;
}

void NvsCudaDesayDsRecv::ResumeReceive() {
    return;
}

void NvsCudaDesayDsRecv::Deinit() {
    std::shared_ptr<sensor_reattach> req_data = std::make_shared<sensor_reattach>();
    std::shared_ptr<sensor_reattach_resp> resq_data = std::make_shared<sensor_reattach_resp>();
    req_data->isattach(false);
    req_data->sensor_id(_sensor_id);
    req_data->index(_channel_id);
    _reattach_clint->RequestAndForget(req_data);
    m_bquit = true;
    _consumer->Stop();
    _reattach_clint->Deinit();
}

int32_t NvsCudaDesayDsRecv::InitNVS(const std::string& ipc_channel) {
    int32_t ret = sscanf(ipc_channel.c_str(), "cam%u_recv%u", &_sensor_id, &_channel_id);
    if (ret != 2) {
        DS_LOG_ERROR << "Invalid ipc channel " << ipc_channel;
        return -1;
    }
    std::shared_ptr<sensor_reattachPubSubType> req_data_type = std::make_shared<sensor_reattachPubSubType>();
    std::shared_ptr<sensor_reattach_respPubSubType> resp_data_type = std::make_shared<sensor_reattach_respPubSubType>();
    std::shared_ptr<sensor_reattach> req_data = std::make_shared<sensor_reattach>();
    std::shared_ptr<sensor_reattach_resp> resq_data = std::make_shared<sensor_reattach_resp>();

    _reattach_clint.reset(new hozon::netaos::cm::Client<sensor_reattach, sensor_reattach_resp>(req_data_type, resp_data_type));
    _reattach_clint->Init(0, "sensor_reattach");
    int isReattachServerOnline;
    do{
        isReattachServerOnline = _reattach_clint->WaitServiceOnline(500);
    } while (isReattachServerOnline != 0);

    SensorInfo sensor_info;
    sensor_info.id = _sensor_id;
    _consumer = std::make_shared<desay::CIpcConsumerChannel>(nv::NVSHelper::GetInstance().sci_buf_module, nv::NVSHelper::GetInstance().sci_sync_module, &sensor_info, desay::CUDA_CONSUMER, _sensor_id,
                                                             _channel_id);

    desay::ConsumerConfig consumer_config{false, false};
    _consumer->SetConsumerConfig(consumer_config);

    auto status = _consumer->CreateBlocks(nullptr);
    if (status != NVSIPL_STATUS_OK) {
        DS_LOG_ERROR << "Fail to create blocks";
        return -1;
    }

    req_data->isattach(true);
    req_data->sensor_id(_sensor_id);
    req_data->index(_channel_id);
    _reattach_clint->RequestAndForget(req_data);
    // std::thread heartsend([this](){
    //     while(!m_bquit){
    //         std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //         std::shared_ptr<sensor_reattach> req_heart = std::make_shared<sensor_reattach>();
    //         req_heart->isalive(true);
    //         req_heart->sensor_id(_sensor_id);
    //         req_heart->index(_channel_id);
    //         _reattach_clint->RequestAndForget(req_heart);
    //     }
    // });
    // heartsend.detach();


    status = _consumer->Connect();
    if (status != NVSIPL_STATUS_OK) {
        DS_LOG_ERROR << "Fail to connect";
        return -1;
    }

    status = _consumer->InitBlocks();
    if (status != NVSIPL_STATUS_OK) {
        DS_LOG_ERROR << "Fail to init blocks";
        return -1;
    }

    status = _consumer->Reconcile();
    if (status != NVSIPL_STATUS_OK) {
        DS_LOG_ERROR << "Fail to reconcile";
        return -1;
    }

    static_cast<desay::CCudaConsumer*>(_consumer->m_upConsumer.get())->SetOnPacketCallback(std::bind(&NvsCudaDesayDsRecv::NVSReadyCallback, this, std::placeholders::_1));
    _consumer->Start();

    DS_LOG_INFO << "Succ to create cuda consumer";
    return 0;
}

void NvsCudaDesayDsRecv::NVSReadyCallback(std::shared_ptr<desay::DesayCUDAPacket> packet) {
    std::shared_ptr<NvsImageCUDA> _nvs_image_cuda(new NvsImageCUDA());

    _nvs_image_cuda->data_time_sec = packet->capture_start_us * 1e-6;
    _nvs_image_cuda->virt_time_sec = 0;
    _nvs_image_cuda->width = packet->width;
    _nvs_image_cuda->height = packet->height;
    _nvs_image_cuda->format = packet->format;
    _nvs_image_cuda->size = packet->data_size;
    _nvs_image_cuda->step = packet->step;
    _nvs_image_cuda->cuda_dev_ptr = packet->cuda_dev_ptr;
    _nvs_image_cuda->need_user_free = packet->need_user_free;
    _nvs_image_cuda->SetReleaseCB(std::bind(&NvsCudaDesayDsRecv::NVSReleaseBufferCB,this,std::placeholders::_1,std::placeholders::_2));

    _nvs_image_cuda->__header.timestamp_real_us = packet->capture_start_us;
    _nvs_image_cuda->__header.timestamp_virt_us = 0;
    _nvs_image_cuda->__header.seq = 0;

    DS_LOG_DEBUG << "check header info: data->__header.timestamp_real_us = " << _nvs_image_cuda->__header.timestamp_real_us;

    int32_t ret = _writer.Write(_nvs_image_cuda);
    if (ret < 0) {
        DS_LOG_ERROR << "Fail to write " << _config.topic;
        return;
    }

    // DS_LOG_INFO << " ---config.topic : " << _config.topic <<"  packet->post_fence : "
    //         <<  _nvs_image_cuda->_packet->post_fence
    //         << " _nvs_image_cuda->cuda_dev_ptr " << _nvs_image_cuda->cuda_dev_ptr << " packet->cuda_dev_ptr " << _nvs_image_cuda->_packet->cuda_dev_ptr
    //         << " need_user_free " << packet->need_user_free << " data_time_sec " << packet->capture_start_us
    //         << " width " << packet->width << " height " << packet->height;
}

void NvsCudaDesayDsRecv::NVSReleaseBufferCB(bool need_free, void* dev_ptr) {
    static_cast<desay::CCudaConsumer*>(_consumer->m_upConsumer.get())->ReleasePacket(need_free,dev_ptr);
}

}  // namespace adf_lite
}  // namespace netaos
}  // namespace hozon

#endif
