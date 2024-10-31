/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Reference.hpp
 */

#ifndef DDS_CORE_REFERENCE_HPP
#define DDS_CORE_REFERENCE_HPP

#include <memory>

namespace dds {
namespace core {
/**
 * @brief  Reference types implement reference semantics.
 */
template<typename Impl>
class Reference {
public:
    /**
     * @brief Creates a new reference to an existing object, increasing the
     * reference count.
     * @param[in] impl Smart pointer of the existing object.
     */
    explicit Reference(std::shared_ptr<Impl> impl) noexcept : refImpl_(std::move(impl))
    {}

    virtual ~Reference() = default;

    /**
     * @brief Returns true only if the referenced object are the same.
     * @param[in] ref Object to Compare.
     * @return bool
     */
    bool operator==(Reference ref) const noexcept
    {
        return refImpl_ == ref.refImpl_;
    }

    /**
     * @brief Returns true only if the referenced object are not the same.
     * @param[in] ref Object to Compare.
     * @return bool
     */
    bool operator!=(Reference ref) const noexcept
    {
        return refImpl_ != ref.refImpl_;
    }

protected:
    std::shared_ptr<Impl> refImpl_;

private:
    void *operator new(size_t memSize) = delete;
};
}
}

#endif /* DDS_CORE_REFERENCE_HPP */

