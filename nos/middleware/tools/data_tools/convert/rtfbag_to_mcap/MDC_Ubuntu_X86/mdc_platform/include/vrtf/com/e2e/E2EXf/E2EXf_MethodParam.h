/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Define Method Input params
 * Create: 2021-07-20
 */

#ifndef VRTF_E2E_E2EXF_METHODPARAM_H
#define VRTF_E2E_E2EXF_METHODPARAM_H
#include <array>
#include <cstdint>
#include <string>
#include <vrtf/com/e2e/E2EXf/E2EXf_MethodType.h>

/* AXIVION disable style AutosarC++19_03-M0.1.4: It's allowed to use tagged unions until C++17 */
namespace vrtf {
namespace com {
namespace e2e {
namespace impl {
struct ServerProtectParaImpl {
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    bool IsSetPara;
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    bool Inplace;
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    MessageResultType MessageResult;
    std::uint16_t ReceivedRequestCounter;
    CSTransactionHandleType CSTransactionHandle;
};

struct ClientCheckParaImpl {
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    bool IsSetPara;
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    bool Inplace;
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    MessageResultType MessageResult;
    std::uint16_t RequestCounter;
};

struct ServerCheckParaImpl {
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    bool IsSetPara;
    /* AXIVION Next Line AutosarC++19_03-M0.1.4: This an external api */
    bool Inplace;
    std::uint16_t RequestCounter;
};
} /* End impl namespace */
} /* End e2e namespace */
} /* End com namespace */
} /* End vrtf namespace */
/* AXIVION enable style AutosarC++19_03-M0.1.4 */
#endif // VRTF_E2E_E2EXF_METHODPARAM_H
