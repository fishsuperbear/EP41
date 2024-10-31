#pragma once

#include <iostream>
#include <functional>
#include <unordered_map>
#include <mutex>

#include "NvSIPLCommon.hpp"
#include "NvSIPLCamera.hpp"
#include "NvSIPLClient.hpp"
#include "NvSIPLPipelineMgr.hpp"
#include "NvSIPLQuery.hpp" // Query
#include "NvSIPLQueryTrace.hpp" // Query Trace
#include "NvSIPLTrace.hpp"

// Sample application header files
#include "cam_utils.hpp"
#include "cam_filewriter.hpp"
#include "image_manager.hpp"
#include "image_queue.hpp"
#include "nv_camera_config.hpp"

namespace hozon {
namespace netaos {
namespace camera {


struct CameraInternalData {
    bool isValid;
    std::string moduleInfo;
    std::vector<uint8_t> data;
};

#pragma pack(push, 1)
struct camera_interal_data_x3f_x8b {
    uint8_t time_year;
    uint8_t time_month;
    uint8_t time_day;
    uint8_t version;
    uint8_t model;
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double k4;
    double k5;
    double k6;
    double p1;
    double p2;
    double averang;
    double maximum;
};

struct camera_interal_data_x031 {
    uint8_t place;
    char line[16];
    char lens_name[16];
    char sensor_name[16];
    char ser_name[16];
    float focus;
    float AA_x;
    float AA_y;
    float cx;
    float cy;
    float fx;
    float fy;
    float xi;
    float k1;
    float k2;
    float k3;
    float p1;
    float p2;
    uint8_t crc_h;
    uint8_t crc_l;
};
#pragma pack(pop)

class ThreadData : public FrameCompleteQueueHandler::ICallback,
                   public NotificationQueueHandler::ICallback
{
public:
    void process(const NvSIPLPipelineNotifier::NotificationData& event);
    void process(INvSIPLClient::INvSIPLBuffer* const & pBuffer);

    SIPLStatus GetSIPLBuffer(std::string& image_data, uint32_t sensor_id);

    std::string threadName;
    NvSIPLPipelineQueues *queues = nullptr;
    FrameCompleteQueueHandler imageQueueHandler;
    NotificationQueueHandler eventQueueHandler;
    std::mutex *printMutex;

    uint32_t uSensorId = MAX_NUM_SENSORS;
    uint32_t uNumFrameDrops = 0U;
    uint32_t uNumFrameDiscontinuities = 0U;

    uint32_t uFrameCounter = 0;

    std::function<void(void)> on_data_frame;
    std::function<int32_t(INvSIPLClient::INvSIPLNvMBuffer* buffer)> nvstream_post;

    INvSIPLClient::INvSIPLNvMBuffer * nvbufptr;
    NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;

    struct Image_Info{
        uint32_t frame_id;
        double time_stamp;
        uint32_t height;
        uint32_t width;
    };
    Image_Info image_info;
};

class NvCameraImpl
{
public:
    NvCameraImpl();
    ~NvCameraImpl();

    SIPLStatus Init(SensorConfig &config);
    SIPLStatus Start();
    SIPLStatus Deinit();

    INvSIPLCamera* GetSIPLCamera();
    std::unordered_map<uint32_t, NvSIPLPipelineConfiguration>& GetSensorInfo();

    using NvStreamCallback = std::function<int32_t(INvSIPLClient::INvSIPLNvMBuffer* buffer)>;
    int32_t RegisterNvStreamPost(NvStreamCallback callback, uint32_t sensor, uint32_t imageCaptureType);

    using FrameAvailableCallback = std::function<void(void)>;
    int32_t RegisterProcess(FrameAvailableCallback callback, uint32_t sensor);

    // TODO.
    SIPLStatus ImageManagerInit();
    void ImageManagerDeInit();
    int32_t GetImageData(std::string& image_data, uint32_t sensor);
    double GetImageTimeStamp(uint32_t sensor);
    uint32_t GetImageHeight(uint32_t sensor);
    uint32_t GetImageWidth(uint32_t sensor);
    uint32_t GetFrameID(uint32_t sensor);

private:
    SIPLStatus InitCameraNito();
    SIPLStatus GetEEPROMData(INvSIPLCamera *siplCamera, uint32_t sensorid);
    int32_t GetCaptureType(uint32_t sensor);

    // sensor instance
    PlatformCfg oPlatformCfg;
    std::unique_ptr<INvSIPLCamera> siplCamera = nullptr;

    NvSIPLDeviceBlockQueues deviceBlockQueues;
    NvSIPLPipelineQueues queues[MAX_NUM_SENSORS]={};
    NvSciBufModule sciBufModule = nullptr;
    NvSciSyncModule sciSyncModule = nullptr;

    std::vector<uint32_t> vSensorIds;
    std::vector<SensorInfo> vSensorInfos;
    std::unordered_map<uint32_t, NvSIPLPipelineConfiguration> vPipelineInfos;

    CImageManager imageManager;
    std::vector<uint8_t> blob;
    bool defaultNitoLoaded = true;
    std::mutex threadPrintMutex;
    ThreadData threadDataStructs[MAX_NUM_SENSORS][THREAD_INDEX_COUNT];

    SensorConfig _config;
};


}
}
}
