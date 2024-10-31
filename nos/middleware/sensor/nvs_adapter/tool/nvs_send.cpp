#include "sensor/nvs_adapter/nvs_adapter_producer.h"
#include "adf-lite/include/sig_stop.h"

using namespace hozon::netaos::nv;
using namespace hozon::netaos;

int main(int argc, char* argv[]) {
    std::string ipc_channel("nvscistream_0");
    std::string producer_name("producer");

    if (argc == 3) {
        ipc_channel = std::string(argv[1]);
        producer_name = std::string(argv[2]);
    }

    hozon::netaos::log::InitLogging(
        "nvs_send",
        "nvs_send",
        hozon::netaos::log::LogLevel::kInfo,
        hozon::netaos::log::HZ_LOG2FILE,
        "/opt/usr/log/soc_log/",
        10,
        20);

    hozon::netaos::adf_lite::SigHandler::GetInstance().Init();

    NVSHelper::GetInstance().Init();

    NvStreamAdapterProducer producer;
    int32_t ret = producer.Init(ipc_channel, producer_name, 3, 1);
    if (ret < 0) {
        NVS_LOG_ERROR << "Fail to init producer.";
        return -1;
    }

    hozon::netaos::adf_lite::SigHandler::GetInstance().NeedStopBlocking();
    NVS_LOG_INFO << "Stop to send.";
}