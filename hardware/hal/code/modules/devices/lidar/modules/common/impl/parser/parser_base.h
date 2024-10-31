#ifndef LIDAR_PARSER_BASE_H
#define LIDAR_PARSER_BASE_H

#include <string.h>
#include <math.h>

#include "lidar/modules/common/impl/utils/lidar_types.h"
#include "lidar/modules/common/hw_lidar_log_impl.h"

class BaseParser
{
public:
    BaseParser() {}
    virtual ~BaseParser() {}

    virtual bool init(const LidarConfig &config) = 0;
    virtual void parse(const Scan &scan, hw_lidar_pointcloud_XYZIT *points) = 0;

protected:
    LidarConfig config_;
};

class RobosenseM1Parser : public BaseParser
{
public:
    RobosenseM1Parser();
    ~RobosenseM1Parser();

    bool init(const LidarConfig &config) override;
    void parse(const Scan &scan, hw_lidar_pointcloud_XYZIT *points) override;

private:
    void unpack(const Scan::Packet &packet, hw_lidar_pointcloud_XYZIT *points);

    int points_size_;
};

class ParserFactory
{
public:
    static BaseParser *createParser(const LidarConfig &config)
    {
        switch (config.lidar_model)
        {
        case HW_LIDAR_MODEL::ROBOSENSE_M1:
            HW_LIDAR_LOG_INFO("create robosense_m1 parser\n");
            return new RobosenseM1Parser();
            break;
        default:
            HW_LIDAR_LOG_ERR("create parser failed! unknown lidar model: %d\n", config.lidar_model);
            return nullptr;
            break;
        }
    }
};

#endif // LIDAR_PARSER_BASE_H