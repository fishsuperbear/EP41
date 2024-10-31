#ifndef NETAOS_CORE_INTERNAL_TYPE_CHECK_H_
#define NETAOS_CORE_INTERNAL_TYPE_CHECK_H_

#include "core/error_code.h"
namespace hozon {

namespace netaos {
namespace core {
template <typename T, typename E>
class Result;

template <typename T, typename E>
class Future;

namespace internal {
template <typename T>
struct IsResult : public std::false_type {};

template <typename T, typename E>
struct IsResult<hozon::netaos::core::Result<T, E>> : public std::true_type {};

template <typename T>
struct IsVoidResult : public std::false_type {};

template <typename E>
struct IsVoidResult<hozon::netaos::core::Result<void, E>> : public std::true_type {};

template <typename T>
struct IsFuture : public std::false_type {};

template <typename T, typename E>
struct IsFuture<hozon::netaos::core::Future<T, E>> : public std::true_type {};

template <typename T>
struct IsVoidFuture : public std::false_type {};

template <typename E>
struct IsVoidFuture<hozon::netaos::core::Future<void, E>> : public std::true_type {};
}  // namespace internal
}  // namespace core
}  // namespace netaos
}  // namespace hozon
#endif
