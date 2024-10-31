#include "lidar/modules/normal/impl/hw_lidar_normal_impl.h"

bool HWLidarNormalContext::running_flag = true;

void signalHandle(int signo)
{
    HW_LIDAR_LOG_INFO("catch interrupt signal: %d\n", signo);
    HWLidarNormalContext::running_flag = false;
}

HWLidarNormalContext::HWLidarNormalContext(struct hw_lidar_t *i_plidar) : HWLidarContext(i_plidar)
{
}

HWLidarNormalContext::~HWLidarNormalContext()
{
    Device_Close();
}

s32 HWLidarNormalContext::Init()
{
    signal(SIGINT, signalHandle);

    return 0;
}

s32 HWLidarNormalContext::Device_Open(struct hw_lidar_callback_t *i_callback)
{
    std::vector<LidarConfig> config_vec = ConfigManager::getInstance()->getLidarConfig(i_callback);
    callback_ = i_callback->data_cb;

    for (int i = 0; i < config_vec.size(); i++)
    {
        std::shared_ptr<ScanCache> p_scancache = std::make_shared<ScanCache>();
        std::string cache_name = "lidar" + std::to_string(i) + " scan cache";
        p_scancache->init(20, cache_name);

        std::thread packet_thread(&HWLidarNormalContext::packetThreadHandle, this, p_scancache, config_vec[i]);
        packet_thread.detach();
        std::thread convert_thread(&HWLidarNormalContext::convertThreadHandle, this, p_scancache, config_vec[i]);
        convert_thread.detach();
    }

    while (running_flag)
    {
        sleep(1);
    }

    return 0;
}

s32 HWLidarNormalContext::Device_Close()
{
    if (callback_)
    {
        callback_ = nullptr;
    }
    running_flag = false;

    return 0;
}

void HWLidarNormalContext::packetThreadHandle(std::shared_ptr<ScanCache> p_scancache, const LidarConfig &config)
{
    SocketProtocol protocol;
    if (!protocol.init(config))
    {
        HW_LIDAR_LOG_ERR("socket protocol init failed!\n");
        return;
    }

    while (running_flag)
    {
        Scan scan;
        if (!poll(config, protocol, scan))
        {
            HW_LIDAR_LOG_ERR("lidar%d poll failed!\n", config.index);
            continue;
        }

        if (scan.packets.empty())
        {
            HW_LIDAR_LOG_ERR("lidar%d data is empty!\n", config.index);
            continue;
        }

        p_scancache->put(scan);
    }
}

void HWLidarNormalContext::convertThreadHandle(const std::shared_ptr<ScanCache> &p_scancache, const LidarConfig &config)
{
    BaseParser *parser = ParserFactory::createParser(config);
    if (!parser->init(config))
    {
        HW_LIDAR_LOG_ERR("robosensem1 parser init failed!\n");
        return;
    }

    while (running_flag)
    {
        Scan scan = p_scancache->take();

        hw_lidar_pointcloud_XYZIT points[config.points_per_frame];
        parser->parse(scan, points);
        
        HW_LIDAR_LOG_INFO("lidar%d send pointcloud, timestamp: %lld\n", config.index, points[config.points_per_frame - 1].timestamp);
        if (callback_)
        {
            callback_(config.index, points, config.points_per_frame);
        }
    }
}

bool HWLidarNormalContext::poll(const LidarConfig &config, SocketProtocol &protocol, Scan &scan)
{
    int packets_per_frame = config.packets_per_frame;
    scan.packets.resize(packets_per_frame);

    for (int i = 0; i < packets_per_frame; i++)
    {
        while (running_flag)
        {
            int res = protocol.getLidarOriginDataRobosenseM1(scan.packets[i]);
            if (res == RECEIVE_SUCCESS)
            {
                break;
            }
            else
            {
                return false;
            }
        }
    }

    return true;
}
