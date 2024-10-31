#ifndef NETAOS_CORE_VARIANT_H
#define NETAOS_CORE_VARIANT_H
#include <stdalign.h>

#include <cfloat>
#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>

#include "core/internal/variant_util.h"
#include "core/utility.h"
namespace hozon {

namespace netaos {
namespace core {
// 23.7.11, class bad_variant_access
class bad_variant_access : public std::exception {
   public:
    bad_variant_access() noexcept = default;

    explicit bad_variant_access(const char* msg) : msg_(msg) {}

    ~bad_variant_access() = default;

    const char* what() const noexcept override { return msg_; }

   private:
    const char* msg_;
};

// 23.7.3, class template variant
template <typename... Types>
class Variant;

// 23.7.4, variant helper classes
template <typename T>
struct variant_size;

template <typename T>
struct variant_size<const T> : variant_size<T> {};

template <typename T>
struct variant_size<volatile T> : variant_size<T> {};

template <typename T>
struct variant_size<const volatile T> : variant_size<T> {};

template <typename T>
constexpr size_t variant_size_v = variant_size<T>::value;

template <typename... Types>
struct variant_size<Variant<Types...>> : std::integral_constant<size_t, sizeof...(Types)> {};

template <size_t I, typename T>
struct variant_alternative;

template <size_t I, typename T>
using variant_alternative_t = typename variant_alternative<I, T>::type;

template <size_t I, typename T>
struct variant_alternative<I, const T> : std::add_const<variant_alternative_t<I, T>> {};

template <size_t I, typename T>
struct variant_alternative<I, volatile T> : std::add_volatile<variant_alternative_t<I, T>> {};

template <size_t I, typename T>
struct variant_alternative<I, const volatile T> : std::add_cv<variant_alternative_t<I, T>> {};

template <size_t I, typename... Types>
struct variant_alternative<I, hozon::netaos::core::Variant<Types...>> {
    static_assert(I < sizeof...(Types), "Index out of bounds in hozon::netaos::core::variant_alternative<>");
    using type = typename hozon::netaos::core::internal::GetTypeAt<0, I, Types...>::type;
};

constexpr size_t variant_npos = static_cast<size_t>(-1);

// 23.7.5, value access
template <typename T, typename... Types>
constexpr bool holds_alternative(const Variant<Types...>& var) noexcept {
    constexpr size_t index = internal::findDetail::FindMatchedIndex<T, Types...>::index;
    static_assert(index != variant_npos - 1, "T occurs more than once in variant alternatives");
    return index == var.index();
}

namespace internal {
template <typename... Types>
void ThrowGetException(const hozon::netaos::core::Variant<Types...>& v) {
    if (v.index() == variant_npos) {
        throw hozon::netaos::core::bad_variant_access{"Variant is valueless"};
    }
    throw hozon::netaos::core::bad_variant_access{"Unexpected index or Unexpected type"};
}
}  // namespace internal

template <size_t I, typename... Types>
constexpr hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>& get(hozon::netaos::core::Variant<Types...>& v) {
    static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
    if (v.index() != I) {
        internal::ThrowGetException(v);
    }
    return *(reinterpret_cast<hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>*>(const_cast<uint8_t*>(v.value_)));
}

template <size_t I, typename... Types>
constexpr hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>&& get(hozon::netaos::core::Variant<Types...>&& v) {
    static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
    if (I != v.index()) {
        internal::ThrowGetException(v);
    }
    return std::move(*(reinterpret_cast<hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>*>(const_cast<uint8_t*>(v.value_))));
}

template <size_t I, typename... Types>
constexpr const hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>& get(const hozon::netaos::core::Variant<Types...>& v) {
    static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
    if (I != v.index()) {
        internal::ThrowGetException(v);
    }
    return *(reinterpret_cast<hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>*>(const_cast<uint8_t*>(v.value_)));
}

template <size_t I, typename... Types>
constexpr const hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>&& get(const hozon::netaos::core::Variant<Types...>&& v) {
    static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
    if (v.index() != I) {
        internal::ThrowGetException(v);
    }
    return std::move(*(reinterpret_cast<hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>*>(const_cast<uint8_t*>(v.value_))));
}

template <typename T, typename... Types>
constexpr T& get(hozon::netaos::core::Variant<Types...>& v) {
    if (!hozon::netaos::core::holds_alternative<T>(v)) {
        internal::ThrowGetException(v);
    }
    return *(reinterpret_cast<T*>(const_cast<uint8_t*>(v.value_)));
}

template <typename T, typename... Types>
constexpr T&& get(hozon::netaos::core::Variant<Types...>&& v) {
    if (!hozon::netaos::core::holds_alternative<T>(v)) {
        internal::ThrowGetException(v);
    }
    return std::move(*(reinterpret_cast<T*>(const_cast<uint8_t*>(v.value_))));
}

template <typename T, typename... Types>
constexpr const T& get(hozon::netaos::core::Variant<Types...> const& v) {
    if (!hozon::netaos::core::holds_alternative<T>(v)) {
        internal::ThrowGetException(v);
    }
    return *(reinterpret_cast<T*>(const_cast<uint8_t*>(v.value_)));
}

template <typename T, typename... Types>
constexpr const T&& get(const hozon::netaos::core::Variant<Types...>&& v) {
    if (!hozon::netaos::core::holds_alternative<T>(v)) {
        internal::ThrowGetException(v);
    }
    return std::move(*(reinterpret_cast<T*>(const_cast<uint8_t*>(v.value_))));
}

template <size_t I, typename... Types>
constexpr std::add_pointer_t<variant_alternative_t<I, Variant<Types...>>> get_if(Variant<Types...>* v) noexcept {
    using AlternativeType = variant_alternative_t<I, Variant<Types...>>;
    static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
    static_assert(!std::is_void<AlternativeType>::value, "Alternative type must not be void");
    if (v && v->index() == I) return std::addressof(get<I>(*v));
    return nullptr;
}

template <size_t I, typename... Types>
constexpr std::add_pointer_t<const variant_alternative_t<I, Variant<Types...>>> get_if(const Variant<Types...>* v) noexcept {
    using AlternativeType = variant_alternative_t<I, Variant<Types...>>;
    static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
    static_assert(!std::is_void<AlternativeType>::value, "Alternative type must not be void");
    if (v && v->index() == I) return std::addressof(get<I>(*v));
    return nullptr;
}

template <typename T, typename... Types>
constexpr std::add_pointer_t<T> get_if(Variant<Types...>* v) noexcept {
    static_assert(!std::is_void<T>::value, "T must not be void");
    constexpr size_t idx = internal::findDetail::FindUnambiguousIndexSfinae<T, Types...>::value;
    return hozon::netaos::core::get_if<idx>(v);
}

template <class T, class... Types>
constexpr std::add_pointer_t<const T> get_if(const Variant<Types...>* v) noexcept {
    static_assert(!std::is_void<T>::value, "T must not be void");
    constexpr size_t idx = internal::findDetail::FindUnambiguousIndexSfinae<T, Types...>::value;
    static_assert(idx != variant_npos, "T is not contained in alternatives");
    return hozon::netaos::core::get_if<idx>(v);
}

template <typename... Types>
class Variant {
    enum : size_t { DataSize = internal::MaxByteSize<Types...>(), AlignSize = internal::MaxAlignSize<Types...>(), FirstAlternativeIndex = 0U };
    using ValueType = typename std::aligned_storage_t<DataSize, AlignSize>;
    using FirstAlternativeType = variant_alternative_t<FirstAlternativeIndex, hozon::netaos::core::Variant<Types...>>;

   public:
    // 23.7.3.1, constructors
    template <typename U = FirstAlternativeType, typename = std::enable_if_t<std::is_default_constructible<U>::value>>
    constexpr Variant() noexcept(std::is_nothrow_default_constructible<U>::value) {
        typeIndex_ = variant_npos;
        new (value_) U();
        typeIndex_ = FirstAlternativeIndex;
    }

    Variant(const Variant& other) {
        if (!other.valueless_by_exception()) {
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<0, Types...>::CopyCtor(other.typeIndex_, other.value_, value_);
            typeIndex_ = other.typeIndex_;
        }
    }

    Variant(Variant&& other) noexcept(internal::typeTraits::conjunction<std::is_nothrow_move_constructible<Types>...>::value) {
        if (!other.valueless_by_exception()) {
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<0, Types...>::MoveCtor(other.typeIndex_, other.value_, value_);
            typeIndex_ = other.typeIndex_;
        }
    }

    template <typename T, typename... Args, size_t I = hozon::netaos::core::internal::findDetail::FindUnambiguousIndexSfinae<T, Types...>::value,
              typename = std::enable_if_t<std::is_constructible<T, Args...>::value>>
    constexpr explicit Variant(hozon::netaos::core::in_place_type_t<T>, Args&&... args) {
        emplace<T>(args...);
    }

    template <typename T, typename U, typename... Args, size_t I = hozon::netaos::core::internal::findDetail::FindUnambiguousIndexSfinae<T, Types...>::value,
              typename = std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args...>::value>>
    constexpr explicit Variant(hozon::netaos::core::in_place_type_t<T>, std::initializer_list<U> il, Args&&... args) {
        emplace<T>(il, args...);
    }

    template <size_t I, typename... Args, typename Ti = variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>, typename = internal::findDetail::ExactlyOneType<Ti, Types...>,
              typename = std::enable_if_t<std::is_constructible<Ti, Args...>::value>>
    constexpr explicit Variant(hozon::netaos::core::in_place_index_t<I>, Args&&... args) {
        emplace<Ti>(args...);
    }

    template <size_t I, typename U, typename... Args, typename Ti = variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>, typename = internal::findDetail::ExactlyOneType<Ti, Types...>,
              typename = std::enable_if_t<std::is_constructible<Ti, std::initializer_list<U>&, Args...>::value>>
    explicit Variant(hozon::netaos::core::in_place_index_t<I>, std::initializer_list<U> il, Args&&... args) {
        emplace<Ti>(il, args...);
    }

    template <typename T, typename U = std::decay_t<T>, typename = std::enable_if_t<!std::is_same<U, Variant>::value>, typename = std::enable_if_t<!hozon::netaos::core::is_inplace_type<U>::value>,
              typename = std::enable_if_t<!hozon::netaos::core::is_inplace_index<U>::value>, size_t I = internal::findDetail::IndexOfConstructedType<Variant, T>::value,
              typename Ti = variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>, typename = std::enable_if_t<std::is_constructible<Ti, T>::value>>
    constexpr Variant(T&& t) noexcept(std::is_nothrow_constructible<Ti, T>::value) : Variant(hozon::netaos::core::in_place_index<I>, std::forward<T>(t)) {}

    // 23.7.3.2, destructor
    ~Variant() {
        if (typeIndex_ != variant_npos) {
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
        }
        typeIndex_ = variant_npos;
    }

    // 23.7.3.3, assignment
    Variant& operator=(const Variant& rhs) {
        if (this == &rhs) {
            return *this;
        }
        if (typeIndex_ == variant_npos && rhs.typeIndex_ == variant_npos) {
            return *this;
        }
        if (rhs.typeIndex_ == variant_npos) {
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
            return *this;
        }
        if (typeIndex_ == rhs.typeIndex_) {
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::CopyAssign(rhs.typeIndex_, rhs.value_, value_);
            typeIndex_ = rhs.typeIndex_;
        } else {
            if (typeIndex_ != variant_npos) {
                internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
            }
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::CopyCtor(rhs.typeIndex_, rhs.value_, value_);
            typeIndex_ = rhs.typeIndex_;
        }
        return *this;
    }

    Variant& operator=(Variant&& rhs) noexcept {
        if (this == &rhs) {
            return *this;
        }
        if (typeIndex_ == variant_npos && rhs.typeIndex_ == variant_npos) {
            return *this;
        }
        if (rhs.typeIndex_ == variant_npos) {
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
            return *this;
        }
        if (typeIndex_ == rhs.typeIndex_) {
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::MoveAssign(rhs.typeIndex_, rhs.value_, value_);
            typeIndex_ = rhs.typeIndex_;
        } else {
            if (typeIndex_ != variant_npos) {
                internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
            }
            typeIndex_ = variant_npos;
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::MoveCtor(rhs.typeIndex_, rhs.value_, value_);
            typeIndex_ = std::move(rhs.typeIndex_);
        }
        return *this;
    }

    template <typename T, typename = std::enable_if_t<!std::is_same<std::decay_t<T>, Variant>::value>,
              size_t I = internal::findDetail::IndexOfConstructedType<hozon::netaos::core::Variant<Types...>, T>::value, typename Ti = variant_alternative_t<I, hozon::netaos::core::Variant<Types...>>,
              typename = std::enable_if_t<std::is_assignable<Ti&, T>::value && std::is_constructible<Ti, T>::value>,
              bool usingEmplace = std::is_nothrow_constructible<Ti, T>::value || !std::is_nothrow_move_constructible<Ti>::value>
    Variant& operator=(T&& t) noexcept(std::is_nothrow_assignable<Ti&, T>::value&& std::is_nothrow_constructible<Ti, T>::value) {
        if (I == typeIndex_) {
            hozon::netaos::core::get<I>(*this) = std::forward<T>(t);
        } else {
            if (usingEmplace) {
                emplace<I>(std::forward<T>(t));
            } else {
                operator=(Variant(std::forward<T>(t)));
            }
        }
        return *this;
    }

    template <typename T, typename... Args, size_t I = hozon::netaos::core::internal::findDetail::FindUnambiguousIndexSfinae<T, Types...>::value,
              typename = std::enable_if_t<std::is_constructible<T, Args...>::value>>
    T& emplace(Args&&... args) {
        if (typeIndex_ != variant_npos) {
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
        }
        typeIndex_ = variant_npos;
        new (value_) T(std::forward<Args>(args)...);
        typeIndex_ = I;
        return *reinterpret_cast<T*>(value_);
    }

    template <typename T, typename U, typename... Args, size_t I = internal::findDetail::FindUnambiguousIndexSfinae<T, Types...>::value,
              typename = std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args...>::value>>
    T& emplace(std::initializer_list<U> il, Args&&... args) {
        if (typeIndex_ != variant_npos) {
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
        }
        typeIndex_ = variant_npos;
        new (value_) T(il, std::forward<Args>(args)...);
        typeIndex_ = I;
        return *reinterpret_cast<T*>(value_);
    }

    template <size_t I, typename... Args, typename T = variant_alternative_t<I, Variant<Types...>>, typename = std::enable_if_t<std::is_constructible<T, Args...>::value>>
    variant_alternative_t<I, Variant>& emplace(Args&&... args) {
        static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
        if (typeIndex_ != variant_npos) {
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
        }
        typeIndex_ = variant_npos;
        new (value_) T(std::forward<Args>(args)...);
        typeIndex_ = I;
        return *reinterpret_cast<T*>(value_);
    }

    template <size_t I, typename U, typename... Args, typename T = variant_alternative_t<I, Variant<Types...>>,
              typename = std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args...>::value>>
    variant_alternative_t<I, Variant>& emplace(std::initializer_list<U> il, Args&&... args) {
        static_assert(I < sizeof...(Types), "The index must be in [0, number of alternatives)");
        if (typeIndex_ != variant_npos) {
            internal::VariantConstructHelper<FirstAlternativeIndex, Types...>::Dtor(typeIndex_, value_);
        }
        typeIndex_ = variant_npos;
        new (value_) T(il, std::forward<Args>(args)...);
        typeIndex_ = I;
        return *reinterpret_cast<T*>(value_);
    }

    // 23.7.3.5, value status
    constexpr bool valueless_by_exception() const noexcept { return typeIndex_ == variant_npos; }

    constexpr size_t index() const noexcept { return typeIndex_; }

    // 23.7.3.6, swap
    void swap(Variant& rhs) noexcept(internal::typeTraits::conjunction<std::is_nothrow_move_constructible<Types>...>::value) {
        static_assert(internal::typeTraits::conjunction<std::is_move_constructible<Types>...>::value, "Types in variant are not all move constructible!");
        if (typeIndex_ == variant_npos || rhs.typeIndex_ == variant_npos) {
            return;
        }
        std::swap(*this, rhs);
    }

   private:
    template <typename FT, typename... FTypes>
    friend constexpr FT& get(hozon::netaos::core::Variant<FTypes...>& var);
    template <typename FT, typename... FTypes>
    friend constexpr FT&& get(hozon::netaos::core::Variant<FTypes...>&& var);
    template <typename FT, typename... FTypes>
    friend constexpr const FT& get(const hozon::netaos::core::Variant<FTypes...>& var);
    template <typename FT, typename... FTypes>
    friend constexpr const FT&& get(const hozon::netaos::core::Variant<FTypes...>&& var);

    template <size_t I, typename... FTypes>
    friend constexpr hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<FTypes...>>& get(hozon::netaos::core::Variant<FTypes...>& v);

    template <size_t I, typename... FTypes>
    friend constexpr hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<FTypes...>>&& get(hozon::netaos::core::Variant<FTypes...>&& v);

    template <size_t I, typename... FTypes>
    friend constexpr const hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<FTypes...>>& get(const hozon::netaos::core::Variant<FTypes...>& v);

    template <size_t I, typename... FTypes>
    friend constexpr const hozon::netaos::core::variant_alternative_t<I, hozon::netaos::core::Variant<FTypes...>>&& get(const hozon::netaos::core::Variant<FTypes...>&& v);

    alignas(AlignSize) uint8_t value_[DataSize]{0U};
    size_t typeIndex_{variant_npos};
};

// 23.7.6, relational operators
namespace internal {
template <typename... Types>
bool RestEqual(hozon::netaos::core::Variant<Types...>, hozon::netaos::core::Variant<Types...>) {
    return false;
}

template <size_t I, typename... Types, typename Ti = variant_alternative_t<I, Variant<Types...>>, typename std::enable_if_t<!std::is_floating_point<std::decay_t<Ti>>::value>* = nullptr>
bool OneEqual(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs) {
    if (lhs.index() == I) {
        return hozon::netaos::core::get<I>(lhs) == hozon::netaos::core::get<I>(rhs);
    }
    return false;
}

template <size_t I, typename... Types, typename Ti = variant_alternative_t<I, Variant<Types...>>, typename std::enable_if_t<std::is_floating_point<std::decay_t<Ti>>::value>* = nullptr>
bool OneEqual(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs) {
    if (lhs.index() == I) {
        return (fabs(static_cast<double>(hozon::netaos::core::get<I>(lhs)) - static_cast<double>(hozon::netaos::core::get<I>(rhs))) < DBL_EPSILON);
    }
    return false;
}

template <size_t I, size_t... Is, typename... Types>
bool RestEqual(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs) {
    return (OneEqual<I>(lhs, rhs) || RestEqual<Is...>(lhs, rhs));
}

template <size_t... Is, typename... Types>
bool equal(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs, std::index_sequence<Is...>) {
    return RestEqual<Is...>(lhs, rhs);
}

template <typename... Types>
bool RestLess(hozon::netaos::core::Variant<Types...>, hozon::netaos::core::Variant<Types...>) {
    return false;
}

template <size_t I, typename... Types>
bool OneLess(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs) {
    if (lhs.index() == I) {
        return hozon::netaos::core::get<I>(lhs) < hozon::netaos::core::get<I>(rhs);
    }
    return false;
}

template <size_t I, size_t... Is, typename... Types>
bool RestLess(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs) {
    return (OneLess<I>(lhs, rhs) || RestLess<Is...>(lhs, rhs));
}

template <size_t... Is, typename... Types>
bool less(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs, std::index_sequence<Is...>) {
    return RestLess<Is...>(lhs, rhs);
}

template <typename... Types>
bool RestGreater(hozon::netaos::core::Variant<Types...>, hozon::netaos::core::Variant<Types...>) {
    return false;
}

template <size_t I, typename... Types>
bool OneGreater(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs) {
    if (lhs.index() == I) {
        return hozon::netaos::core::get<I>(lhs) > hozon::netaos::core::get<I>(rhs);
    }
    return false;
}

template <size_t I, size_t... Is, typename... Types>
bool RestGreater(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs) {
    return (OneGreater<I>(lhs, rhs) || RestGreater<Is...>(lhs, rhs));
}

template <size_t... Is, typename... Types>
bool greater(hozon::netaos::core::Variant<Types...> lhs, hozon::netaos::core::Variant<Types...> rhs, std::index_sequence<Is...>) {
    return RestGreater<Is...>(lhs, rhs);
}
}  // namespace internal

template <typename... Types>
constexpr bool operator==(const Variant<Types...>& v, const Variant<Types...>& w) {
    if (v.index() != w.index()) {
        return false;
    }
    if (v.valueless_by_exception()) {
        return true;
    }
    return internal::equal(v, w, std::make_index_sequence<sizeof...(Types)>{});
}

template <typename... Types>
constexpr bool operator!=(const Variant<Types...>& v, const Variant<Types...>& w) {
    if (v.index() != w.index()) {
        return true;
    }
    if (v.valueless_by_exception()) {
        return false;
    }
    return !internal::equal(v, w, std::make_index_sequence<sizeof...(Types)>{});
}

template <typename... Types>
constexpr bool operator<(const Variant<Types...>& v, const Variant<Types...>& w) {
    if (w.valueless_by_exception()) {
        return false;
    }
    if (v.valueless_by_exception()) {
        return true;
    }
    if (v.index() < w.index()) {
        return true;
    }
    if (v.index() > w.index()) {
        return false;
    }
    return internal::less(v, w, std::make_index_sequence<sizeof...(Types)>{});
}

template <typename... Types>
constexpr bool operator>(const Variant<Types...>& v, const Variant<Types...>& w) {
    if (v.valueless_by_exception()) {
        return false;
    }
    if (w.valueless_by_exception()) {
        return true;
    }
    if (v.index() > w.index()) {
        return true;
    }
    if (v.index() < w.index()) {
        return false;
    }
    return internal::greater(v, w, std::make_index_sequence<sizeof...(Types)>{});
}

template <typename... Types>
constexpr bool operator<=(const Variant<Types...>& v, const Variant<Types...>& w) {
    if (v.valueless_by_exception()) {
        return true;
    }
    if (w.valueless_by_exception()) {
        return false;
    }
    if (v.index() < w.index()) {
        return true;
    }
    if (v.index() > w.index()) {
        return false;
    }
    return !internal::greater(v, w, std::make_index_sequence<sizeof...(Types)>{});
}

template <typename... Types>
constexpr bool operator>=(const Variant<Types...>& v, const Variant<Types...>& w) {
    if (w.valueless_by_exception()) {
        return true;
    }
    if (v.valueless_by_exception()) {
        return false;
    }
    if (v.index() > w.index()) {
        return true;
    }
    if (v.index() < w.index()) {
        return false;
    }
    return !internal::less(v, w, std::make_index_sequence<sizeof...(Types)>{});
}

// 23.7.8, class monostate
struct monostate {};

// 23.7.9, monostate relational operators
constexpr bool operator<(hozon::netaos::core::monostate, hozon::netaos::core::monostate) noexcept { return false; }

constexpr bool operator>(hozon::netaos::core::monostate, hozon::netaos::core::monostate) noexcept { return false; }

constexpr bool operator<=(hozon::netaos::core::monostate, hozon::netaos::core::monostate) noexcept { return true; }

constexpr bool operator>=(hozon::netaos::core::monostate, hozon::netaos::core::monostate) noexcept { return true; }

constexpr bool operator==(hozon::netaos::core::monostate, hozon::netaos::core::monostate) noexcept { return true; }

constexpr bool operator!=(hozon::netaos::core::monostate, hozon::netaos::core::monostate) noexcept { return false; }

// 23.7.10, specialized algorithms
template <typename... Types, typename = std::enable_if_t<(internal::typeTraits::conjunction<std::is_move_constructible<Types>...>::value)>>
void swap(Variant<Types...>& v, Variant<Types...>& w) noexcept(internal::typeTraits::conjunction<std::is_nothrow_move_constructible<Types>...>::value) {
    v.swap(w);
}
}  // namespace core
}  // namespace netaos
}  // namespace hozon
// 23.7.12, hash support
namespace std {
template <>
struct hash<hozon::netaos::core::monostate> {
    size_t operator()(const hozon::netaos::core::monostate&) const {
        return 66740831;  // The value is same with iAOS-V200R007C10B010 clang version 10.0.1
    }
};
}  // namespace std
#endif
