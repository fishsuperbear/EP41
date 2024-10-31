/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: 序列化框架类型定义
 * Author: s00515168
 */

#ifndef ARA_PER_SERIALIZER_SERIALIZER_TYPE_H
#define ARA_PER_SERIALIZER_SERIALIZER_TYPE_H

#include <netinet/in.h>
#include <type_traits>
#include "ara/per/per_base_type.h"
#include "ara/per/kvs_type.h"

namespace ara {
namespace per {
inline bool IsLittleEndian()
{
    union {
        int32_t a;
        char_t b;
    } c;
    c.a = 1;
    return (c.b == 1);
}

template<typename T>
T Htonl64(const T& host)
{
    T result = host;
    if (IsLittleEndian()) {
        std::uint32_t low = host & 0xFFFFFFFFU;
        std::uint32_t high = (host >> 32U) & 0xFFFFFFFFU;
        low = htonl(low);
        high = htonl(high);

        result = low;
        result <<= 32U;
        result |= high;
    }
    return result;
}

template<typename T>
T Ntohl64(const T& host)
{
    T result = host;
    if (IsLittleEndian()) {
        std::uint32_t low = static_cast<std::uint32_t>(host & 0xFFFFFFFFU);
        std::uint32_t high = static_cast<std::uint32_t>((host >> 32U) & 0xFFFFFFFFU);
        low = ntohl(low);
        high = ntohl(high);

        result = low;
        result <<= 32U;
        result |= static_cast<std::uint64_t>(high);
    }
    return result;
}

template <typename T, typename Tagged = void>
struct is_enumerable
{
    static const bool value = false;
};

template <typename T>
struct is_enumerable<T, typename T::IsEnumerableTag>
{
    static const bool value = true;
};

template<typename T>
using is_string = std::is_same<T, ara::core::String>;

template<typename T>
using is_kvtype = std::is_same<T, ara::per::kvstype::KvsType>;

template <typename T>
constexpr bool IsSerializable()
{
    return std::is_scalar<T>::value || is_enumerable<T>::value || is_string<T>::value;
}

template <typename T>
using EnableIfScalar = typename std::enable_if<std::is_scalar<T>::value>::type;

template <typename T>
using EnableIfEnumerable = typename std::enable_if<is_enumerable<T>::value>::type;

template<typename T>
using EnableIfBase = typename std::enable_if<std::is_scalar<T>::value ||
    is_string<T>::value || is_kvtype<T>::value>::type;

template<typename T>
using EnableIfCustom = typename std::enable_if<!(
    std::is_scalar<T>::value || is_string<T>::value || is_kvtype<T>::value)>::type;
}  // namespace per
}  // namespace ara
#endif
