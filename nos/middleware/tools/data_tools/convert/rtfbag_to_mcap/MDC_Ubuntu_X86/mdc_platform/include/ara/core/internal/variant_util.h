/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: the implementation of Promise class according to AutoSAR standard core type
 * Create: 2021-12-18
 */
#ifndef ARA_CORE_VARIANT_UTIL_H
#define ARA_CORE_VARIANT_UTIL_H
#include <type_traits>
#include <limits>

namespace ara {
namespace core {
template <typename... Types>
class Variant;
namespace internal {
namespace typeTraits {
template <typename... Types>
struct VoidType {
    using type = void;
};

template <typename... Types>
using void_t = typename VoidType<Types...>::type;

template<typename>
constexpr size_t zero_v = 0;

template <typename...>
struct conjunction : public std::true_type {};

template <typename C1>
struct conjunction<C1> : public C1 {};

template <typename C1, typename C2>
struct conjunction<C1, C2> : public std::conditional<C1::value, C2, C1>::type {};

template <typename C1, typename... Cs>
struct conjunction<C1, Cs...> : public std::conditional<C1::value, conjunction<Cs...>, C1>::type {};
}
namespace findDetail {
enum : size_t {
    NotFound = std::numeric_limits<size_t>::max(),
    Ambiguous = NotFound - 1
};

template <typename T, typename... Types>
inline constexpr size_t FindIndex() {
    constexpr bool matches[] = {std::is_same<T, Types>::value...};
    size_t result = NotFound;
    for (size_t i = 0; i < sizeof...(Types); ++i) {
        if (matches[i]) {
            if (result != NotFound) {
              return Ambiguous;
            }
            result = i;
        }
    }
    return result;
}

template <size_t idx>
struct FindUnambiguousIndexSfinaeImpl
    : std::integral_constant<size_t, idx> {};

template <>
struct FindUnambiguousIndexSfinaeImpl<NotFound> {
};

template <>
struct FindUnambiguousIndexSfinaeImpl<Ambiguous> {
};

template <typename T, typename... Types>
struct FindUnambiguousIndexSfinae
    : FindUnambiguousIndexSfinaeImpl<FindIndex<T, Types...>()> {};


inline constexpr size_t FindIdxReturn(size_t currIdx, size_t res, bool match) {
    return !match ? res : (res == NotFound ? currIdx : Ambiguous);
}

template <size_t N>
inline constexpr size_t FindIdx(size_t i, const bool (&matches)[N]) {
    return i == N ? NotFound : FindIdxReturn(i, FindIdx(i + 1, matches), matches[i]);
}

template <typename T, typename... Args>
struct FindMatchedIndex {
    static constexpr bool matches[sizeof...(Args)] = {std::is_same<T, Args>::value...};
    static constexpr size_t index = FindIdx(0, matches);
};

template <typename T>
struct FindMatchedIndex<T> {
    static_assert(!std::is_same<T, T>::value, "T is not in empty alternatives");
};

template <typename T, typename... Args>
struct ExactlyOneImpl : public FindMatchedIndex<T, Args...> {
    static_assert(FindMatchedIndex<T, Args...>::index != NotFound, "T is not contained in alternatives");
    static_assert(FindMatchedIndex<T, Args...>::index != Ambiguous, "T occurs more than once in alternatives");
    using type = T;
};

template <typename T, typename... Args>
using ExactlyOneType = typename ExactlyOneImpl<T, Args...>::type;

template <typename Variant, size_t I = 0>
struct ImaginaryFun;

template <std::size_t I>
struct ImaginaryFun<ara::core::Variant<>, I> {
  static void Run() = delete;
};

template <typename T, typename... Types, size_t I>
struct ImaginaryFun<ara::core::Variant<T, Types...>, I> : ImaginaryFun<ara::core::Variant<Types...>, I + 1> {
    using ImaginaryFun<ara::core::Variant<Types...>, I + 1>::Run;

    static std::integral_constant<size_t, I> Run(const T&, std::integral_constant<size_t, I>);
    static std::integral_constant<size_t, I> Run(T&&, std::integral_constant<size_t, I>);
};

template <typename Variant, typename T, typename = void>
struct IndexOfConstructedType {};

template <typename Variant, typename T>
struct IndexOfConstructedType<Variant, T,
    ara::core::internal::typeTraits::void_t<decltype(ImaginaryFun<Variant>::Run(std::declval<T>(), {}))>>
    : decltype(ImaginaryFun<Variant>::Run(std::declval<T>(), {})) {};

} // namespace findDetail

template <typename T>
inline constexpr T MaxSize(const T& size) noexcept
{
    return size;
}

template <typename T>
inline constexpr T MaxSize(const T& lhs, const T& rhs) noexcept
{
    return (rhs < lhs) ? lhs : rhs;
}

template <typename T, typename... Types>
inline constexpr T MaxSize(const T& lhs, const T& rhs, const Types&... args) noexcept
{
    return MaxSize(MaxSize(lhs, rhs), args...);
}

template <typename... Types>
inline constexpr size_t MaxByteSize() noexcept
{
    return MaxSize(sizeof(Types)...);
}

template <typename... Types>
inline constexpr size_t MaxAlignSize() noexcept
{
    return MaxSize(alignof(Types)...);
}

template <size_t N, typename T, typename U, typename... Types>
struct GetTypeIndex
{
    enum : size_t {
        index = GetTypeIndex<N + 1, T, Types...>::index
    };
};

template <size_t N, typename T, typename... Types>
struct GetTypeIndex<N, T, T, Types...>
{
    enum : size_t {
        index = N
    };
};

template <size_t N, size_t Index, typename T, typename... Types>
struct GetTypeAt
{
    using type = typename GetTypeAt<N + 1, Index, Types...>::type;
};

template <size_t N, typename T, typename... Types>
struct GetTypeAt<N, N, T, Types...>
{
    using type = T;
};

template <size_t N, typename Type, typename... Types>
struct VariantConstructHelper {

static void CopyCtor(const size_t index, const uint8_t *src, uint8_t *dest) noexcept
{
    if (N == index) {
        new(dest) Type (*reinterpret_cast<Type* >(const_cast<uint8_t*>(src)));
    } else {
        VariantConstructHelper<N + 1, Types...>::CopyCtor(index, src, dest);
    }
}

static void MoveCtor(const size_t index, const uint8_t *src, uint8_t *dest) noexcept
{
    if (N == index) {
        new(dest) Type (std::move(*reinterpret_cast<Type* >(const_cast<uint8_t*>(src))));
    } else {
        VariantConstructHelper<N + 1, Types...>::MoveCtor(index, src, dest);
    }
}

static void CopyAssign(const size_t index, const uint8_t *src, uint8_t *dest) noexcept
{
    if (N == index) {
        *reinterpret_cast<Type*>(dest) = *reinterpret_cast<Type*>(const_cast<uint8_t*>(src));
    } else {
        VariantConstructHelper<N + 1, Types...>::CopyAssign(index, src, dest);
    }
}

static void MoveAssign(const size_t index, const uint8_t *src, uint8_t *dest) noexcept
{
    if (N == index) {
        *reinterpret_cast<Type*>(dest) = std::move(*reinterpret_cast<Type*>(const_cast<uint8_t*>(src)));
    } else {
        VariantConstructHelper<N + 1, Types...>::MoveAssign(index, src, dest);
    }
}

static void Dtor(const size_t index, uint8_t* ptr) noexcept
{
    if (N == index) {
        reinterpret_cast<Type*>(ptr)->~Type();
    } else {
        VariantConstructHelper<N + 1, Types...>::Dtor(index, ptr);
    }
}
};

template <size_t N, typename Type>
struct VariantConstructHelper<N, Type> {

static void CopyCtor(const size_t index, const uint8_t* src, uint8_t* dest) noexcept
{
    if (N == index) {
        new (dest) Type (*reinterpret_cast<Type* >(const_cast<uint8_t*>(src)));
    }
}

static void MoveCtor(const size_t index, const uint8_t* src, uint8_t* dest) noexcept
{
    if (N == index) {
        new(dest) Type (std::move(*reinterpret_cast<Type* >(const_cast<uint8_t*>(src))));
    }
}

static void CopyAssign(const size_t index, const uint8_t *src, uint8_t *dest) noexcept
{
    if (N == index) {
        *reinterpret_cast<Type*>(dest) = *reinterpret_cast<Type*>(const_cast<uint8_t*>(src));
    }
}

static void MoveAssign(const size_t index, const uint8_t *src, uint8_t *dest) noexcept
{
    if (N == index) {
        *reinterpret_cast<Type*>(dest) = std::move(*reinterpret_cast<Type*>(const_cast<uint8_t*>(src)));
    }
}

static void Dtor(const size_t index, uint8_t* ptr) noexcept
{
    if (N == index) {
        reinterpret_cast<Type*>(ptr)->~Type();
    }
}
};
}
}
}
#endif
