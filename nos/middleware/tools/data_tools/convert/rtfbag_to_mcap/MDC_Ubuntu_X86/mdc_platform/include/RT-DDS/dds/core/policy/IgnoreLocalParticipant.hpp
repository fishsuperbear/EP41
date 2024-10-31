/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: IgnoreLocalParticipant.hpp
 */

#ifndef DDS_CORE_POLICY_IGNORE_LOCAL_PARTICIPANT_HPP
#define DDS_CORE_POLICY_IGNORE_LOCAL_PARTICIPANT_HPP

namespace dds {
namespace core {
namespace policy {
class IgnoreLocalParticipant {
public:
    IgnoreLocalParticipant() = default;

    ~IgnoreLocalParticipant() = default;

    void Value(bool v) noexcept
    {
        value_ = v;
    }

    bool Value() const noexcept
    {
        return value_;
    }

private:
    bool value_{false};
};
}
}
}

#endif /* DDS_CORE_POLICY_IGNORE_LOCAL_PARTICIPANT_HPP */

