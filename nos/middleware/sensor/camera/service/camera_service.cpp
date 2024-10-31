#include <iostream>
#include <memory>
#include <unistd.h>
#include <csignal>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "sensor/camera/include/nv_camera_impl.hpp"
#include "log/include/default_logger.h"
#include "sensor/camera/service/nvs_sender.h"

using namespace hozon::netaos;
using namespace hozon::netaos::camera;
using namespace hozon::netaos::nv;

bool g_stopFlag = false;
std::mutex mtx;
std::condition_variable cv;

void INTSigHandler(int32_t num)
{
    (void)num;

    g_stopFlag = true;
    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_all();
}

int main(int argc, char* argv[]) {
    /*Need add SIGTERM from EM*/
    signal(SIGTERM, INTSigHandler);
    signal(SIGINT, INTSigHandler);
    signal(SIGPIPE, SIG_IGN);

    DefaultLogger::GetInstance().InitLogger();
    DF_LOG_INFO << "Start..";

    std::shared_ptr<NvCameraImpl> nv_camera = std::make_shared<NvCameraImpl>();
    SensorConfig sensor_config {
        .platform_name = MULTI,
        .pipeline {
            {
                .sensor_name = SENSOR_ISX031,
                .captureOutputRequested = true,
                .isp0OutputRequested = false,
                .isp1OutputRequested = false,
                .isp2OutputRequested = false,
            },
            {
                .sensor_name = SENSOR_0X8B40,
                .captureOutputRequested = false,
                .isp0OutputRequested = true,
                .isp1OutputRequested = false,
                .isp2OutputRequested = false,
            },
            {
                .sensor_name = SENSOR_0X03F,
                .captureOutputRequested = false,
                .isp0OutputRequested = true,
                .isp1OutputRequested = false,
                .isp2OutputRequested = false,
            }
        },
        .vMasks = multi_mask_List,
    };

    nv_camera->Init(sensor_config);
    INvSIPLCamera* siplCamera = nv_camera->GetSIPLCamera();
    if (siplCamera == nullptr) {
        DF_LOG_INFO << "siplCamera is null";
    }

    std::unordered_map<uint32_t, NvSIPLPipelineConfiguration> vPipelineInfos = nv_camera->GetSensorInfo();

    for (auto &it : vPipelineInfos) {
        DF_LOG_INFO << " sensor id : " << it.first 
            << " captureOutputRequested : " << it.second.captureOutputRequested
            << " isp0OutputRequested : " << it.second.isp0OutputRequested
            << " isp1OutputRequested : " << it.second.isp1OutputRequested
            << " isp2OutputRequested : " << it.second.isp2OutputRequested;
    }
    // nv_camera->ImageManagerInit();

    NVSHelper::GetInstance().Init();
    // NVSEventService::GetInstance().Init();
    std::unordered_map<uint32_t, std::shared_ptr<NVSSender>> nvs_senders;
    int32_t ret = 0;
    for (auto &it : vPipelineInfos) {
        uint32_t sensor_id = it.first;
        uint32_t num_consumers = atoi(argv[1]); // TODO: read num_consumers from config and check validity
        std::vector<std::string> ipc_channels;
        ipc_channels.resize(num_consumers);
        for (uint32_t i = 0; i < num_consumers; ++i) {
            ipc_channels[i] = std::string("cam") + std::to_string(sensor_id) + "_send" + std::to_string(i);
        }

        INvSIPLClient::ConsumerDesc::OutputType image_capture_type;

        if (it.second.captureOutputRequested == true) {
            image_capture_type = INvSIPLClient::ConsumerDesc::OutputType::ICP;
        }
        else if (it.second.isp0OutputRequested == true) {
            image_capture_type = INvSIPLClient::ConsumerDesc::OutputType::ISP0;
        }
        else if (it.second.isp1OutputRequested == true) {
            image_capture_type = INvSIPLClient::ConsumerDesc::OutputType::ISP1;
        }
        else if (it.second.isp2OutputRequested == true) {
            image_capture_type = INvSIPLClient::ConsumerDesc::OutputType::ISP2;
        }
        else {
            DF_LOG_ERROR << "Inavlid output type\n";
            return -1;
        }

        std::shared_ptr<NVSSender> sender = std::make_shared<NVSSender>();
        ret = sender->Init(ipc_channels, 
            std::string("Camera") + std::to_string(sensor_id), 
            6, 
            num_consumers, 
            image_capture_type,
            sensor_id,
            siplCamera);
        if (ret < 0) {
            return -1;
        }
        nv_camera->RegisterNvStreamPost(
            std::bind(&NVSBlockSIPLProducer::Post, &sender->_nvs_sipl_producer, std::placeholders::_1), 
            sensor_id, 
            (uint32_t)image_capture_type);
        nvs_senders[sensor_id] = sender;
    }

    // NVSEventService::GetInstance().Run();
    for (auto& sender : nvs_senders) {
        DF_LOG_INFO << "Wait sender ready.";
        while (!sender.second->_nvs_sipl_producer.Ready() && !g_stopFlag) {
            sleep(1);
        }
    }
    DF_LOG_INFO << "All sender ready.";

    nv_camera->Start();    

    while (!g_stopFlag) {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck);
        break;
    }

    nv_camera->Deinit();
    for (auto& sender : nvs_senders) {
        sender.second->Deinit();
    }
    // nv_camera->ImageManagerDeInit();

    return 0;
}