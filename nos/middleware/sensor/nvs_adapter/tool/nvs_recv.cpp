#include "sensor/nvs_adapter/nvs_adapter_consumer.h"
#include "adf-lite/include/sig_stop.h"
#include "adf/include/simple_freq_checker.h"
#include <fstream>

using namespace hozon::netaos::nv;
using namespace hozon::netaos;
using namespace hozon::netaos::adf;

uint32_t sensor_id;
uint32_t channel_id;
std::shared_ptr<SimpleFreqChecker> _freq_checker;

void WriteFile(const std::string& name, uint8_t* data, uint32_t size) {
    std::ofstream of(name);

    if (!of) {
        NVS_LOG_ERROR << "Fail to open " << name;
        return;
    }

    of.write((const char*)data, size);
    of.close();
    NVS_LOG_INFO << "Succ to write " << name;

    return;
}

uint64_t GetMonotonicNs() {
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

void PacketRecvCallback(NvStreamAdapterConsumer* consumer, ImageCUDAPacket* packet) {
    uint8_t* local_ptr = (uint8_t*)malloc(packet->data_size);

    /* Instruct CUDA to copy the packet data buffer to the target buffer */
    uint32_t cuda_rt_err = cudaMemcpy(local_ptr,
                                packet->cuda_dev_ptr,
                                packet->data_size,
                                cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuda_rt_err) {
        NVS_LOG_CRITICAL << "Failed to issue copy command, ret " << log::loghex((uint32_t)cuda_rt_err);
        return;
    }

    // NVS_LOG_INFO << "Recv packet version: " << packet->metadata_local_ptr->version
    //     << ", start: " << packet->metadata_local_ptr->capture_start_us
    //     << ", end: " << packet->metadata_local_ptr->capture_end_us
    //     << ", mono: " << GetMonotonicNs();
    _freq_checker->say(std::string("cam") + std::to_string(sensor_id));
    static int index = 0;
    ++index;
    if ((index > 10) && (index < 14)) {
        WriteFile(std::string("cam") + std::to_string(sensor_id) + "_" + std::to_string(index) + ".yuv", local_ptr, packet->data_size);
    }

    if (packet->need_user_free) {
        cudaFree(packet->cuda_dev_ptr);
    }
    consumer->_nvs_img_consumer.ReleasePacket(packet);
    free(local_ptr);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "nvs_recv sensor_id channel_id\n";
        return 0;
    }

    sensor_id = atoi(argv[1]);
    channel_id = atoi(argv[2]);

    std::string ipc_channel = std::string("cam") + std::to_string(sensor_id) + "_recv" + std::to_string(channel_id);
    std::string consumer_name = "nvs_recv consumer";

    hozon::netaos::log::InitLogging(
        "nvs_recv",
        "nvs_recv",
        hozon::netaos::log::LogLevel::kInfo,
        hozon::netaos::log::HZ_LOG2FILE,
        "/opt/usr/log/soc_log/",
        10,
        20);

    NVS_LOG_INFO << "Init ipc channel " << ipc_channel;
     _freq_checker = std::make_shared<SimpleFreqChecker>([](const std::string& name, double freq){
        NVS_LOG_INFO << "Check " << name << " frequency: " << freq << " Hz";
     });

    hozon::netaos::adf_lite::SigHandler::GetInstance().Init();

    NVSHelper::GetInstance().Init();

    NvStreamAdapterConsumer consumer;
    int32_t ret = consumer.Init(ipc_channel, consumer_name, std::bind(&PacketRecvCallback, &consumer, std::placeholders::_1));
    if (ret < 0) {
        NVS_LOG_ERROR << "Fail to init consumer.";
        return -1;
    }

    hozon::netaos::adf_lite::SigHandler::GetInstance().NeedStopBlocking();
    consumer.Deinit();
    NVS_LOG_INFO << "Stop to recv.";
}