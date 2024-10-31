/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip def
 */

#ifndef DOSOMEIP_FUNCTIONS_H
#define DOSOMEIP_FUNCTIONS_H

#include <stdint.h>
#include "DoSomeIPStubImpl.hpp"
#include "diag/dosomeip/common/dosomeip_def.h"
#include "diag/common/include/to_string.h"

void PrintRequest(const v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage& _req);
void PrintRequest(const hozon::netaos::diag::DoSomeIPReqUdsMessage& req);

void PrintResponse(const v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& resp);
void PrintResponse(const hozon::netaos::diag::DoSomeIPRespUdsMessage& resp);

bool ConvertStruct(const v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage& _req, hozon::netaos::diag::DoSomeIPReqUdsMessage& req);
bool ConvertStruct(const hozon::netaos::diag::DoSomeIPRespUdsMessage& resp, v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& _resp);

std::string GetAddressTypeString(const hozon::netaos::diag::TargetAddressType& type);

#endif  // DOSOMEIP_FUNCTIONS_H