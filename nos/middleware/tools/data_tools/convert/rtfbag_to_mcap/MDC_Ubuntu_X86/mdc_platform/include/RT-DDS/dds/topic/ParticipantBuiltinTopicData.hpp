/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: ParticipantBuiltinTopicData.hpp
 */

#ifndef DDS_TOPIC_PARTICIPANT_BUILTIN_TOPIC_DATA_HPP
#define DDS_TOPIC_PARTICIPANT_BUILTIN_TOPIC_DATA_HPP

#include <RT-DDS/dds/core/Types.hpp>
#include <RT-DDS/dds/core/policy/UserData.hpp>
#include <RT-DDS/dds/topic/BuiltinTopicKey.hpp>

namespace dds {
namespace topic {
class ParticipantBuiltinTopicData {
public:
    dds::topic::BuiltinTopicKey Key() const noexcept
    {
        return key_;
    }

    const dds::core::policy::UserData &UserData() const noexcept
    {
        return userData_;
    }

private:
    dds::topic::BuiltinTopicKey key_{};
    dds::core::policy::UserData userData_{};
    friend class dds::sub::ParticipantBuiltinDataReaderImpl;
};

using ParticipantBuiltinTopicDataSeq = std::vector<ParticipantBuiltinTopicData>;
}
}

#endif /* DDS_TOPIC_PARTICIPANT_BUILTIN_TOPIC_DATA_HPP */

