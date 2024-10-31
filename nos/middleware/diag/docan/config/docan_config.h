/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanConfig Header
 */

#ifndef DOCAN_CONFIG_H_
#define DOCAN_CONFIG_H_

#include <stdint.h>
#include <mutex>
#include <vector>
#include <string>
#include "diag/docan/common/docan_internal_def.h"

namespace hozon {
namespace netaos {
namespace diag {


class DocanConfig {
public:
    static DocanConfig *instance();
    static void destroy();
    virtual ~DocanConfig();

    int32_t Init();
    int32_t Start();
    int32_t Stop();
    int32_t Deinit();

    int32_t loadConfig();

    uint16_t getEcu(uint16_t canid_rx);

    bool getEcuInfo(const uint16_t ecu, N_EcuInfo_t& info);

    const std::vector<N_EcuInfo_t>& getEcuInfoList();

    const std::vector<N_RouteInfo_t>& getRouteInfoList();

private:
    DocanConfig(const DocanConfig &);
    DocanConfig & operator = (const DocanConfig &);

    DocanConfig();

private:
    static std::mutex instance_mtx_;
    static DocanConfig *instancePtr_;

    std::vector<N_EcuInfo_t>    can_ecu_info_list_;
    std::vector<N_RouteInfo_t>  can_route_info_list_;
};

} // end of diag
} // end of netaos
} // end of hozon

#endif  // DOCAN_CONFIG_H_
/* EOF */