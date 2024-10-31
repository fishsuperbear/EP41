/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Help definitions related to template derivation
 * Create: 2021-07-20
 */
#ifndef VRTF_VCC_UTILS_TEMPLATE_DEDUCTION_HELPER_H
#define VRTF_VCC_UTILS_TEMPLATE_DEDUCTION_HELPER_H
#include <memory>
#include "ara/core/array.h"
#include "ara/core/optional.h"
namespace vrtf {
namespace vcc {
namespace utils {
namespace TemplateDeduction {
// shared_ptr helper
template<typename T> struct IsSharedPtr : public std::false_type {};
template<typename T> struct IsSharedPtr<std::shared_ptr<T>> : public std::true_type {};
// array helper
template<typename T> struct IsArray : public std::false_type {};
template<typename T, std::size_t N> struct IsArray<std::array<T, N>> : public std::true_type {};
// ap cm struct helper
template <typename T, typename Tag = void> struct IsStruct { static const bool value = false; };
template <typename T> struct IsStruct<T, typename T::IsEnumerableTag> { static const bool value = true; };
// ap cm Optional helper
template <typename T> struct IsOptional : public std::false_type {};
template <typename T> struct IsOptional<ara::core::Optional<T>> : public std::true_type {};
// variadic template(args) helper
template<std::size_t I, class... AArgs>
class Parameters;
template<std::size_t I, class Head, class... PArgs>
class Parameters<I, Head, PArgs...> {
public:
    using Type = typename Parameters<I - 1U, PArgs...>::Type;
};
template<class Head>
class Parameters<0U, Head> {
public:
    using Type = Head;
};
template<class Head, class... PArgs>
class Parameters<0U, Head, PArgs...> {
public:
    using Type = Head;
};
}
}
}
}
#endif
