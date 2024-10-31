/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的共享访问类
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_PER_INCLUDE_UNIQUE_HANDLE_H_
#define MIDDLEWARE_PER_INCLUDE_UNIQUE_HANDLE_H_

#include <memory>
#include <utility>
namespace hozon {
namespace netaos {
namespace per {
template <typename T>
class UniqueHandle final {
 public:
    UniqueHandle() = default;

    explicit UniqueHandle(std::unique_ptr<T>&& objectValue) : object_(std::move(objectValue)) {}

    UniqueHandle(UniqueHandle const& other) = delete;

    UniqueHandle(UniqueHandle&& other) = default;

    UniqueHandle& operator=(UniqueHandle const& other) = delete;

    UniqueHandle& operator=(UniqueHandle&& other) & = default;

    T* operator->() const noexcept { return object_.get(); }

    T& operator*() const noexcept { return *object_.get(); }

    explicit operator bool() const noexcept { return !!object_; }

    std::unique_ptr<T> GetValue() noexcept { return std::move(object_); }

    ~UniqueHandle() = default;

 private:
    std::unique_ptr<T> object_;
};
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_UNIQUE_HANDLE_H_
