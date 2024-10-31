#pragma once

#include "framework/concurrency/detail/atomic_shared_ptr.h"

namespace netaos {
namespace framework {
namespace concurrency {

template <typename T>
using AtomicSharedPtr = folly::atomic_shared_ptr<T>;

template <typename T>
using SharedPtr = std::shared_ptr<T>;

} // namespace concurrency
} // namespace framework
} // namespace netaos
