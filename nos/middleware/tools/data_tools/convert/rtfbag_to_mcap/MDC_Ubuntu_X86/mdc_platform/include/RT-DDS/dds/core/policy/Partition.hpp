/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Partition.hpp
 */

#ifndef DDS_CORE_POLICY_PARTITION_HPP
#define DDS_CORE_POLICY_PARTITION_HPP

#include <RT-DDS/dds/core/Types.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Set of strings that introduces a logical partition among the topics
 * visible by a dds::pub::Publisher and a dds::sub::Subscriber.
 */
class Partition {
public:
    /**
     * @brief Creates a policy with the default partition.
     */
    Partition() = default;

    ~Partition() = default;

    /**
     * @brief Creates a policy with a single partition with the specified name.
     */
    explicit Partition(
        StringSeq name)
        : name_(name)
    {}

    /**
     * @brief Sets the partition names specified in a vector.
     */
    void Name(StringSeq name) noexcept
    {
        name_ = std::move(name);
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const StringSeq &Name() const noexcept
    {
        return name_;
    }

    /**
     * @brief A tool fun to convert this to a readable string
     * @return a copy of readable string
     */
    std::string ToString() const;

private:
    StringSeq name_{};
};
}
}
}

#endif /* DDS_CORE_POLICY_PARTITION_HPP */

