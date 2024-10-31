/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: BuiltinTopicKey.hpp
 */

#ifndef DDS_TOPIC_BUILTIN_TOPIC_KEY_HPP
#define DDS_TOPIC_BUILTIN_TOPIC_KEY_HPP

#include <cstdint>
#include <array>

namespace dds {
namespace sub {
class ParticipantBuiltinDataReaderImpl;
}
}

namespace dds {
namespace topic {
class BuiltinTopicKey {
public:
    using ValueType = std::array<int32_t, 4U>;   /* Array of 4 int32_t */

    ValueType Value() const noexcept
    {
        return value_;
    }
private:
    ValueType value_{};
    friend class dds::sub::ParticipantBuiltinDataReaderImpl;
};
}
}

#endif /* DDS_TOPIC_BUILTIN_TOPIC_KEY_HPP */

