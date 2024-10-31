/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Define types in communication mannger
 * Create: 2019-07-24
*/
#ifndef VRTF_VCC_API_INTERNAL_METHOD_ERROR_H
#define VRTF_VCC_API_INTERNAL_METHOD_ERROR_H
#include <cstdint>
namespace vrtf {
namespace vcc {
namespace api {
namespace types {
struct MethodError {
    uint64_t domainValue = 0U;
    int32_t errorCode = 0;
};
}
}
}
}
#endif
