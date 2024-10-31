/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: diag client socket
 */
#include "uds_request_someip.h"
#include "common.h"


int32_t
UdsResqestSomeip::SomeIPActivate(doip_payload_t *requestInfo)
{
    return 0;
}

int32_t
UdsResqestSomeip::SomeIPRequest(doip_payload_t *requestInfo)
{
    return 0;
}

int32_t UdsResqestSomeip::SomeIPResponse(doip_payload_t *msg_type)
{
    return 0;
}

int32_t UdsResqestSomeip::StartTestUdsSomeIP()
{
    std::cout << "##uds on someip" << std::endl;

    return 0;
}

