/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CanGateway Header
 */


#ifndef DOCAN_GATEWAY_H_
#define DOCAN_GATEWAY_H_

#include <linux/can.h>
#include <linux/can/raw.h>
#include <linux/netlink.h>

#include "diag/docan/common/docan_internal_def.h"

namespace hozon {
namespace netaos {
namespace diag {


class CanGateway
{
public:
    CanGateway(N_RouteInfo_t routeInfo);
    virtual ~CanGateway();

    int32_t Init();
    int32_t Start();
    int32_t Stop();

    N_RouteInfo_t GetRouteInfo();

    bool IsStop();

    int32_t AddRouteRule(const std::string& fromDevice, const uint16_t fromCanid, const std::string& forwardDevice, const uint16_t fowardCanid);
    int32_t DelRouteRule(const std::string& fromDevice, const uint16_t fromCanid, const std::string& forwardDevice, const uint16_t fowardCanid);

private:
    CanGateway(const CanGateway&);
    CanGateway& operator=(const CanGateway&);

    int32_t FlushRouteRule(const std::string& fromDevice);

    int32_t addattr_l(struct nlmsghdr *n, int32_t maxlen, int32_t type, const void *data, int32_t alen);

private:
    int32_t m_sock = -1;
    bool stopFlag_ = false;

    N_RouteInfo_t route_info_;

};


} // end of diag
} // end of netaos
} // end of hozon

#endif  // DOCAN_GATEWAY_H_
