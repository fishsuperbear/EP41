/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file optional.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_OPTIONAL_H_
#define APD_ARA_CORE_OPTIONAL_H_

#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

#include "ara/core/exception.h"
#include "ara/core/utility.h"

namespace ara {
namespace core {
inline namespace _19_11 {

/// optional for object types
template <class T>
class Optional;

/// Disengaged state indicator
struct nullopt_t {
    struct placeholder {};
    explicit constexpr nullopt_t( placeholder ) {}
};

constexpr nullopt_t nullopt{ nullopt_t::placeholder() };

namespace  // unnamed
{
template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<Optional<T>> : std::true_type {};

template <typename U, typename T>
using EnableIfConvertible = typename std::enable_if<std::is_convertible<U, T>::value>::type;
template <typename T>
using EnableIfCopyConstructible =
    typename std::enable_if<std::is_copy_constructible<T>::value>::type;
template <typename T>
using EnableIfMoveConstructible =
    typename std::enable_if<std::is_move_constructible<T>::value>::type;
template <typename T>
using EnableIfNotOptional =
    typename std::enable_if<!is_optional<typename std::decay<T>::type>::value>::type;
template <typename T>
using EnableIfLValueReference = typename std::enable_if<std::is_lvalue_reference<T>::value>::type;
template <typename T>
using EnableIfNotLValueReference =
    typename std::enable_if<!std::is_lvalue_reference<T>::value>::type;
template <typename T, typename... Args>
using EnableIfConstructible =
    typename std::enable_if<std::is_constructible<T, Args...>::value>::type;
}  // unnamed namespace

/**
 * @brief A overload of std::optional.
 *
 * @tparam T the type of val in the this Optional class
 *
 * @uptrace{SWS_CORE_01033}
 */
template <class T>
class Optional {
   public:
    typedef T value_type;

    // X.Y.4.1, constructors
    /**
     * @brief Default constructor
     *
     * @uptrace{SWS_CORE_01034}
     */
    constexpr Optional() noexcept : init( false ), val( nullptr ) {}

    /**
     * @brief construct from nullopt_t object. This constructor is same  as default constructor
     * Optional().
     *
     * @param nullopt_t [in] object of struct nullopt
     */
    constexpr Optional( nullopt_t ) noexcept : init( false ), val( nullptr ) {}
    // Delete the predefined copy-constructor (for volatile, since we don't need
    // it).
    // Doing so prevents another copy constructor from being implicitly declared.
    // This will allow us to SFINAE-in the copy-constructor when needed.
    // Optional( const volatile Optional & ) = delete;

    // template <typename T_ = T, typename = EnableIfCopyConstructible<T_> >
    /**
     * @brief construct from another instance.
     *
     * @param other [in] the other instance
     *
     * @uptrace{SWS_CORE_01036}
     */
    constexpr Optional( const Optional<T> &other )
        : init( other.init ), val( other.init ? new T( *other.val ) : nullptr ) {}
    // Delete the predefined move-constructor (for volatile, since we don't need
    // it).
    // Doing so prevents another move constructor from being implicitly declared.
    // This will allow us to SFINAE-in the move-constructor when needed.
    // Optional( volatile Optional &&rh ) = delete;

    // template <typename T_ = T, typename = EnableIfMoveConstructible<T_> >
    /**
     * @brief Move construct from another instance.
     *
     * @param other  the other instance
     *
     * @uptrace{SWS_CORE_01035}
     */
    Optional( Optional<T> &&other ) noexcept( std::is_nothrow_move_constructible<T>::value )
        : init( other.init ), val( std::move( other.val ) ) {}
    // Delete the predefined copy-constructor (for volatile, since we don't need
    // it).
    // Doing so prevents another copy constructor from being implicitly declared.
    // This will allow us to SFINAE-in the copy-constructor when needed.
    Optional( const volatile T & ) = delete;

    // template <typename T_ = T, typename = EnableIfCopyConstructible<T_> >
    /**
     * @brief construct from val
     *
     * @param v [in] the value that is assigned to the val in this Optional class
     *
     */
    constexpr Optional( const T &v ) : init( true ), val( new T( v ) ) {}
    // Delete the predefined move-constructor (for volatile, since we don't need
    // it).
    // Doing so prevents another move constructor from being implicitly declared.
    // This will allow us to SFINAE-in the move-constructor when needed.
    // Optional( volatile T &&t ) = delete;
    // template <typename T_ = T, typename = EnableIfMoveConstructible<T_> >
    /**
     * @brief Move construct from val
     *
     * @param v [in] it is assigned to the val in this Optional class
     *
     */
    constexpr Optional( T &&v ) : init( true ), val( new T( std::move( v ) ) ) {}

    /**
     * @brief construct from parameters used to construct val
     *
     * @tparam args parameters used to construct val
     */
    template <class... Args>
    constexpr explicit Optional( in_place_t, Args &&... args )
        : init( true ), val( new T( std::forward<Args>( args )... ) ) {}

    /**
     * @brief construct from parameters used to construct class T
     *
     * @tparam U the type of the std::initializer_list used to construct class T
     * @tparam Args the type of the parameters used to construct class T
     * @param ilist [in] the std::initializer_list used to construct class T
     * @param args [in] the parameters used to construct class T
     */
    template <class U, class... Args>
    constexpr explicit Optional( in_place_t, std::initializer_list<U> ilist, Args &&... args )
        : init( true ), val( new T( ilist, std::forward<Args>( args )... ) ) {}
    // X.Y.4.2, destructor
    /**
     * @brief destructor
     *
     * @uptrace{SWS_CORE_01037}
     */
    ~Optional() = default;

    // X.Y.4.3, assignment
    /**
     * @brief assign a null Optional object to current instance
     *
     * @param nullopt_t null Optional object
     * @return *this
     */
    Optional<T> &operator=( nullopt_t ) noexcept {
        init = false;
        val.reset( nullptr );
        return *this;
    }
    /**
     * @brief Copy assign from another instance.
     *
     * @param rhs [in] the other instance
     * @return *this
     *
     * @uptrace{SWS_CORE_01039}
     */
    Optional<T> &operator=( const Optional<T> &rhs ) {
        init = rhs.init;
        val.reset( rhs.init ? new T( *rhs.val ) : nullptr );
        return *this;
    }
    /**
     * @brief Move assign from another instance.
     *
     * @param rhs [in] the other instance
     * @return *this
     *
     * @uptrace{SWS_CORE_01038}
     */
    Optional<T> &operator=( Optional<T> &&rhs ) noexcept(
        std::is_nothrow_move_assignable<T>::value &&std::is_nothrow_move_constructible<T>::value ) {
        init = rhs.init;
        val.reset( rhs.init ? new T( std::move( *rhs.val ) ) : nullptr );
        return *this;
    }
    /**
     * @brief Move assign from parameters used to construct T.
     *
     * @tparam U the type of the parameters used to construct T.
     * @param rhs the parameters used to construct T.
     * @return *this
     *
     * @uptrace{SWS_CORE_01038}
     */
    template <class U, typename = EnableIfNotOptional<U>>
    Optional<T> &operator=( U &&rhs ) {
        init = true;
        val.reset( new T( std::move( rhs ) ) );
        return *this;
    }

    /**
     * @brief emplace the current instance with parameters used to construct T.
     *
     * @tparam Args the type of the parameters used to construct T.
     * @param args [in] the parameters used to construct T.
     */
    template <class... Args>
    T& emplace( Args &&... args ) {
        *this = nullopt;
        init  = true;
        val.reset( new T( std::forward<Args>( args )... ) );
        return *val;
    }

    /**
     * @brief emplace the current instance with initializer_list and parameters used to construct T.
     *
     * @tparam U the type of the initializer_list's elements used to construct T.
     * @tparam Args the type of the parameters used to construct T.
     * @param il [in] the the initializer_list used to construct T.
     * @param args [in] the parameters used to construct T.
     */
    template <class U, class... Args>
    T& emplace( std::initializer_list<U> il, Args &&... args ) {
        *this = nullopt;
        init  = true;
        val.reset( new T( il, std::forward<Args>( args )... ) );
        return *val;
    }
    /**
     * @brief reset the current instance with default values
     *
     * @uptrace{SWS_CORE_01043}
     */
    void reset() noexcept {
        init = false;
        val.reset( nullptr );
    }

    // X.Y.4.4, swap
    /**
     * @brief Add overload of swap for ara::core::Optional
     *
     * @tparam T_ the type of val in Optional class.
     * @remark This method shall not participate in overload resolution unless: T has move
     * constructor.
     * @param rhs [in] the other instance.
     */
    template <typename T_ = T, typename = EnableIfMoveConstructible<T_>>
    void swap( Optional<T> &rhs ) noexcept( std::is_nothrow_move_constructible<T>::value &&noexcept(
        std::swap( std::declval<T &>(), std::declval<T &>() ) ) ) {
        std::swap( this->val, rhs.val );
        std::swap( init, rhs.init );
    }

    // X.Y.4.5, observers
    /**
     * @brief Access the contained value.
     *
     * @return a pointer to the contained value
     */
    constexpr T const *operator->() const { return val.get(); }
    /**
     * @brief Access the contained value.
     *
     * @return a const pointer to the contained value
     */
    T *operator->() { return val.get(); }
    /**
     * @brief Access the contained value.
     *
     * @return a const reference of the contained value
     */
    constexpr T const &operator*() const { return *val; }
    /**
     * @brief Access the contained value.
     *
     * @return a reference of the contained value
     */
    T &operator*() { return *val; }
    /**
     * @brief Check whether *this contains a value
     *
     * @return true if *this contains a value, false otherwise
     *
     * @uptrace{SWS_CORE_01041}
     */
    constexpr bool has_value() const noexcept { return init; }
    /**
     * @brief Check whether *this contains a value.
     *
     * @return true if *this contains a value, false otherwise
     *
     * @uptrace{SWS_CORE_01042}
     */
    constexpr explicit operator bool() const noexcept { return init; }

   private:
    bool               init;
    std::unique_ptr<T> val;
};

// Relational operators
/**
 * @brief Global operator== for Optional instances.
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param y [in] the right hand side of the comparison
 * @return true if the two instances compare equal, false otherwise
 */
template <class T>
constexpr bool operator==( const Optional<T> &x, const Optional<T> &y ) {
    return bool( x ) != bool( y ) ? false : ( !bool( x ) ? true : *x == *y );
}

/**
 * @brief Global operator!= for Optional instances.
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param y [in] the right hand side of the comparison
 * @return true if the two instances compare unequal, false otherwise
 */
template <class T>
constexpr bool operator!=( const Optional<T> &x, const Optional<T> &y ) {
    return !( x == y );
}

/**
 * @brief Global operator< for Optional instances
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param y [in] the right hand side of the comparison
 * @return true in case this Optional is lexically less than other, false else.
 */
template <class T>
constexpr bool operator<( const Optional<T> &x, const Optional<T> &y ) {
    return !bool( y ) ? false : ( !bool( x ) ? true : std::less<T>{}( *x, *y ) );
}

/**
 * @brief Global operator> for Optional instances
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param y [in] the right hand side of the comparison
 * @return true in case this Optional is lexically more than other, false else.
 */
template <class T>
constexpr bool operator>( const Optional<T> &x, const Optional<T> &y ) {
    return !( x < y ) && !( x == y );
}

/**
 * @brief Global operator<= for Optional instances
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param y [in] the right hand side of the comparison
 * @return true in case this Optional is lexically less than or equal to other, false else.
 */
template <class T>
constexpr bool operator<=( const Optional<T> &x, const Optional<T> &y ) {
    return ( x < y ) || ( x == y );
}

/**
 * @brief Global operator>= for Optional instances
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param y [in] the right hand side of the comparison
 * @return true in case this Optional is lexically more than or equal to other, false else.
 */
template <class T>
constexpr bool operator>=( const Optional<T> &x, const Optional<T> &y ) {
    return !( x < y );
}

/**
 * @brief Equal comparison with nullopt
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param nullopt_t the right hand side of the comparison
 * @return true in case this Optional is lexically equal to nullopt_t, false else.
 */
template <class T>
constexpr bool operator==( const Optional<T> &x, nullopt_t ) noexcept {
    return !bool( x );
}

/**
 * @brief Equal comparison with nullopt
 *
 * @tparam T the type of class Optional's val
 * @param nullopt_t the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case this Optional is lexically equal to nullopt_t, false else.
 */
template <class T>
constexpr bool operator==( nullopt_t, const Optional<T> &x ) noexcept {
    return !bool( x );
}

/**
 * @brief Nequal comparison with nullopt
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param nullopt_t the right hand side of the comparison
 * @return true in case this Optional is lexically nequal to nullopt_t, false else.
 */
template <class T>
constexpr bool operator!=( const Optional<T> &x, nullopt_t ) noexcept {
    return !( x == nullopt );
}

/**
 * @brief Nequal comparison with nullopt
 *
 * @tparam T the type of class Optional's val
 * @param nullopt_t the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case this Optional is lexically nequal to nullopt_t, false else.
 */
template <class T>
constexpr bool operator!=( nullopt_t, const Optional<T> &x ) noexcept {
    return !( nullopt == x );
}

/**
 * @brief Equal comparison with T
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param v [in] the right hand side of the comparison
 * @return true in case the val of this Optional x is not null and lexically equal to v, false else.
 */
template <class T>
constexpr bool operator==( const Optional<T> &x, const T &v ) {
    return bool( x ) ? *x == v : false;
}

/**
 * @brief Equal comparison with T
 *
 * @tparam T the type of class Optional's val
 * @param v [in] the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case the val of this Optional x is not null and lexically equal to v, false else.
 */
template <class T>
constexpr bool operator==( const T &v, const Optional<T> &x ) {
    return bool( x ) ? v == *x : false;
}

/**
 * @brief Nequal comparison with T
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param v [in] the right hand side of the comparison
 * @return true in case the Optional x is nequal to v, false else.
 */
template <class T>
constexpr bool operator!=( const Optional<T> &x, const T &v ) {
    return !( x == v );
}

/**
 * @brief Nequal comparison with T
 *
 * @tparam T the type of class Optional's val
 * @param v [in] the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case the Optional x is nequal to v, false else.
 */
template <class T>
constexpr bool operator!=( const T &v, const Optional<T> &x ) {
    return !( v == x );
}

/**
 * @brief Global operator< for Optional instance and T
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param v [in] the right hand side of the comparison
 * @return true in case this Optional is lexically more than or equal to other, false else.
 */
template <class T>
constexpr bool operator<( const Optional<T> &x, const T &v ) {
    return bool( x ) ? std::less<T>{}( *x, v ) : true;
}

/**
 * @brief Global operator< for T and Optional instance
 *
 * @tparam T the type of class Optional's val
 * @param v [in] the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case the val of the Optional x is not null and
 * v is lexically less than the val of Optional x, false else.
 */
template <class T>
constexpr bool operator<( const T &v, const Optional<T> &x ) {
    return bool( x ) ? std::less<T>{}( v, *x ) : false;
}

/**
 * @brief Global operator<= for Optional instance and T
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param v [in] the right hand side of the comparison
 * @return true in case the val of the Optional x is equal to or less than
 * the v, false else.
 */
template <class T>
constexpr bool operator<=( const Optional<T> &x, const T &v ) {
    return ( x < v ) || ( x == v );
}

/**
 * @brief Global operator<= for T and Optional instance
 *
 * @tparam T the type of class Optional's val
 * @param v [in] the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case v is less than or equal to the val of the Optional x
 * false else.
 */
template <class T>
constexpr bool operator<=( const T &v, const Optional<T> &x ) {
    return ( v < x ) || ( v == x );
}

/**
 * @brief Global operator> for Optional instance and T
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param v [in] the right hand side of the comparison
 * @return true in case the val of the Optional x is more than v,
 * false else.
 */
template <class T>
constexpr bool operator>( const Optional<T> &x, const T &v ) {
    return !( x < v ) && !( x == v );
}

/**
 * @brief Global operator> for T and Optional instance
 *
 * @tparam T the type of class Optional's val
 * @param v [in] the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case v is more than the val of the Optional x,
 * false else.
 */
template <class T>
constexpr bool operator>( const T &v, const Optional<T> &x ) {
    return !( v < x ) && !( v == x );
}

/**
 * @brief Global operator>= for Optional instance and T
 *
 * @tparam T the type of class Optional's val
 * @param x [in] the left hand side of the comparison
 * @param v [in] the right hand side of the comparison
 * @return true in case the val of Optional x is more than or equal to the v,
 * false else.
 */
template <class T>
constexpr bool operator>=( const Optional<T> &x, const T &v ) {
    return !( x < v );
}

/**
 * @brief Global operator>= for T and Optional instance
 *
 * @tparam T the type of class Optional's val
 * @param v [in] the left hand side of the comparison
 * @param x [in] the right hand side of the comparison
 * @return true in case v is more than or equal to the val of Optional x,
 * false else.
 */
template <class T>
constexpr bool operator>=( const T &v, const Optional<T> &x ) {
    return !( v < x );
}

// Specialized algorithms
/**
 * @brief Add overload of std::swap for Optional.
 *
 * @tparam T  the type of val in the Optional
 * @param x  [in] the first argument of the swap invocation
 * @param y  [in] the second argument of the swap invocation
 */
template <class T>
void swap( Optional<T> &x, Optional<T> &y ) noexcept( noexcept( x.swap( y ) ) ) {
    x.swap( y );
}

/**
 * @brief construct from rvalue_reference of T
 *
 * @tparam T  the type of val in the Optional
 * @remark This method shall not participate in overload resolution unless: T is not
 * lvalue_reference.
 * @param v [in] the value that is used to construct the Optional instance.
 * @return the new Optional instance.
 */
template <class T, typename = EnableIfNotLValueReference<T>>
constexpr Optional<typename std::decay<T>::type> make_optional( T &&v ) {
    return Optional<typename std::decay<T>::type>( std::forward<T>( v ) );
}

/**
 * @brief construct from lvalue_reference of T
 *
 * @tparam T  the type of val in the Optional
 * @remark This method shall not participate in overload resolution unless: T is lvalue_reference.
 * @param v  [in] the value that is used to construct the Optional instance.
 * @return the new Optional instance.
 */
template <class T, typename = EnableIfLValueReference<T>>
constexpr Optional<typename std::decay<T>::type> make_optional(
    typename std::remove_reference<T>::type &&v ) {
    return Optional<typename std::decay<T>::type>(
        std::forward<typename std::remove_reference<T>::type>( v ) );
}

// hash support
template <class T>
struct hash;
/**
 * @brief A struct that privide the method to get the hash value of the Optional instance.
 *
 * @tparam T the type of val in the this Optional class
 *
 * @uptrace{SWS_CORE_01033}
 */
template <class T>
struct hash<Optional<T>> {
    /**
     * @brief get the hash value of the Optional instance.
     *
     * @param o [in] the Optional instance
     * @return the hash value of the Optional instance
     */
    std::size_t operator()( const Optional<T> &o ) const {
        if ( bool( o ) ) {
            return std::hash<T>()( *o );
        } else {
            return 0;
        }
        return 0;
    }
};
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_OPTIONAL_H_
/* EOF */
