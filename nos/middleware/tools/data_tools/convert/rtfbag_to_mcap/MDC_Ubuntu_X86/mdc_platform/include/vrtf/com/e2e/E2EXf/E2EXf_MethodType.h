/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Define Method basic type
 * Create: 2021-07-20
 */
#ifndef VRTF_E2E_E2EXF_METHODTYPE_H
#define VRTF_E2E_E2EXF_METHODTYPE_H

#include <cstdint>

namespace vrtf {
namespace com {
namespace e2e {
namespace impl {
/* --------------------------------------- Type definitions for Method protection ---------------------------------- */
struct CSTransactionHandleType {
    std::uint32_t Counter;
    std::uint32_t SourceID;
};

enum class MethodPortPrototype : std::uint8_t {
    /* Server */
    RPortPrototype = 0,
    /* Client */
    PPortPrototype = 1
};

enum class MessageResultType : std::uint8_t {
    MESSAGERESULT_OK = 0,
    MESSAGERESULT_ERROR = 1
};

enum class MessageTypeType : std::uint8_t {
    MESSAGETYPE_REQUEST = 0,
    MESSAGETYPE_RESPONSE = 1
};

struct E2EMethodParams {
    MethodPortPrototype PortPrototype;
    MessageResultType MessgaeResult;
    MessageTypeType MessageType;
    bool IsInCorrectID;
    std::uint32_t IncorrectId;
};
} /* End namespace impl */
} /* End namesapce e2e */
} /* End namesapce com */
} /* End namespace vrtf */
#endif // VRTF_E2E_E2EXF_METHODTYPE_H
