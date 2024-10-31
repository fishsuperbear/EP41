/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: function_group header
 * Create: 2020-05-18
 */
#ifndef VRTF_FUNCTION_GROUP_H
#define VRTF_FUNCTION_GROUP_H

#include <ara/core/future.h>
#include "exec_error_domain.h"

namespace ara {
namespace exec {
class FunctionGroup {
public:
    class CtorToken {
    public:
        explicit CtorToken(core::StringView const &functionGroupName) : functionGroupName_(functionGroupName) {};
        ~CtorToken() = default;
        core::StringView functionGroupName_;
    };

    static core::Result<FunctionGroup::CtorToken> Preconstruct(core::StringView metaModelIdentifier) noexcept;
    FunctionGroup(FunctionGroup::CtorToken &&token) noexcept;
    bool operator == (FunctionGroup const &other) const noexcept;
    bool operator != (FunctionGroup const &other) const noexcept;
    ~FunctionGroup() noexcept = default;

    core::StringView functionGroupName_;
};
}
}

#endif // VRTF_FUNCTION_GROUP_STATE_H

