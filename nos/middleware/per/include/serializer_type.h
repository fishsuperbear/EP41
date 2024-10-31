/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块序列化和反序列化实现接口
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_SERIALIZER_TYPE_H_
#define MIDDLEWARE_PER_INCLUDE_SERIALIZER_TYPE_H_

#include <netinet/in.h>

#include <string>
#include <type_traits>

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace hozon {
namespace netaos {
namespace per {

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
using is_proto = std::is_same<T, google::protobuf::Message>;
template <typename T>
constexpr bool IsSerializable() {
    return std::is_arithmetic<T>::value || is_enumerable<T>::value || is_string<T>::value;
}

template <typename T>
using EnableIfArithmetic = typename std::enable_if<std::is_arithmetic<T>::value>::type;

template <typename T>
using EnableIfEnumerable = typename std::enable_if<is_enumerable<T>::value>::type;

template <typename T>
using EnableIfBase = typename std::enable_if<std::is_arithmetic<T>::value || is_string<T>::value || is_proto<T>::value>::type;
template <typename T>
using EnableIfCustom = typename std::enable_if<!(std::is_arithmetic<T>::value || is_string<T>::value || is_proto<T>::value)>::type;

}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_SERIALIZER_TYPE_H_
