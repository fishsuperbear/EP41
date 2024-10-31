/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: cfg
 * Description: typename类型定义
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_CFG_INCLUDE_SERIALIZER_TYPE_H_
#define MIDDLEWARE_CFG_INCLUDE_SERIALIZER_TYPE_H_

#include <netinet/in.h>

#include <string>
#include <type_traits>

namespace hozon {
namespace netaos {
namespace cfg {

template <typename T, typename Tagged = void>
struct is_enumerable {
    static const bool value = false;
};

template <typename T>
struct is_enumerable<T, typename T::IsEnumerableTag> {
    static const bool value = true;
};

template <typename T>
using is_string = std::is_same<T, std::string>;

template <typename T>
constexpr bool IsSerializable() {
    return std::is_arithmetic<T>::value || is_enumerable<T>::value || is_string<T>::value;
}

template <typename T>
using EnableIfArithmetic = typename std::enable_if<std::is_arithmetic<T>::value>::type;

template <typename T>
using EnableIfEnumerable = typename std::enable_if<is_enumerable<T>::value>::type;

template <typename T>
using EnableIfBase = typename std::enable_if<std::is_arithmetic<T>::value || is_string<T>::value>::type;
template <typename T>
using EnableIfCustom = typename std::enable_if<!(std::is_arithmetic<T>::value || is_string<T>::value)>::type;

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_CFG_INCLUDE_SERIALIZER_TYPE_H_
