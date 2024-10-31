/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的非共享访问类
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_PER_INCLUDE_SHARED_HANDLE_H_
#define MIDDLEWARE_PER_INCLUDE_SHARED_HANDLE_H_

#include <memory>
#include <utility>
namespace hozon {
namespace netaos {
namespace per {
template <typename T>
class SharedHandle final {
 public:
    SharedHandle() = default;

    explicit SharedHandle(std::shared_ptr<T>&& objectValue) : object_(std::move(objectValue)) {}

    SharedHandle(SharedHandle const& other) = delete;

    SharedHandle(SharedHandle&& other) = default;

    SharedHandle& operator=(SharedHandle const& other) = delete;

    SharedHandle& operator=(SharedHandle&& other) & = default;

    T* operator->() const noexcept { return object_.get(); }

    T& operator*() const noexcept { return *object_.get(); }

    explicit operator bool() const noexcept { return !!object_; }

    std::shared_ptr<T> GetValue() noexcept { return std::move(object_); }

    ~SharedHandle() = default;

 private:
    std::shared_ptr<T> object_;
};
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_SHARED_HANDLE_H_
