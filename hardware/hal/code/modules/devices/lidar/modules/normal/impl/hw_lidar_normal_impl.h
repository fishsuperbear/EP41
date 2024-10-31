#ifndef HW_LIDAR_SOCKET_ROBOSENSENM1_IMPL_H
#define HW_LIDAR_SOCKET_ROBOSENSENM1_IMPL_H

#include "lidar/modules/common/impl/hw_lidar_context_impl.h"
#include "lidar/modules/normal/common/impl/hw_lidar_normal_common_impl.h"

s32 hw_module_privapi_init(void** io_ppvoid);
s32 hw_module_privapi_check_device_api_version(u32 i_device_api_version);
s32 hw_module_privapi_trigger_check_unmasklogleft();
s32 hw_module_privapi_device_get(struct hw_module_t* i_pmodule, void* i_param, struct hw_device_t** io_ppdevice);
s32 hw_module_privapi_device_put(struct hw_module_t* i_pmodule, struct hw_device_t* i_pdevice);

class HWLidarNormalContext : public HWLidarContext
{
public:
    HWLidarNormalContext(struct hw_lidar_t *i_plidar);
    virtual ~HWLidarNormalContext();

    s32 Init() override;
    s32 Device_Open(struct hw_lidar_callback_t *i_callback) override;
    s32 Device_Close() override;

    static bool running_flag;

private:
    bool poll(const LidarConfig &config, SocketProtocol &protocol, Scan &scan);

    void packetThreadHandle(std::shared_ptr<ScanCache> p_scancache, const LidarConfig &config);
    void convertThreadHandle(const std::shared_ptr<ScanCache> &p_scancache, const LidarConfig &config);

    hw_lidar_pointcloud_callback callback_;
};

class LidarNormalEnv
{
public:
    LidarNormalEnv() {}
    ~LidarNormalEnv() {}

    u32 device_api_version_ = 0;

    HWLidarNormalContext *pcontext = nullptr;
};

#endif // HW_LIDAR_SOCKET_ROBOSENSENM1_IMPL_H
