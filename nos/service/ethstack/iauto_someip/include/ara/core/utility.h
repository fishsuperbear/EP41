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
 * @file utility.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_UTILITY_H_
#define APD_ARA_CORE_UTILITY_H_

#include <climits>
#include <cstddef>
#include <initializer_list>
#include <iterator>

namespace ara {
namespace core {
inline namespace _19_11 {
namespace uti_internal {

#ifdef __GNUC__
#ifndef ATTR_PACKED
#define APD_PACKED __attribute__( ( packed ) )
#else
// Do here what is necessary for achieving SWS_CORE_10101 for this compiler!
#define APD_PACKED
#endif
#endif

/// @brief A non-integral binary type
///
/// This is a class-based implementation. It fulfils all the requirements
/// but also allows other, non-conforming uses, in particular initialization
/// with brackets, e.g. "Byte b(42)". This is not possible with a
/// std::byte-based implementation and thus should not be used.
///
/// It is implementation-defined whether this type can be used for type aliasing
/// without triggering Undefined Behavior.
/// If in doubt, instruct your compiler to exhibit non-strict aliasing behavior
/// with something like gcc's -fno-strict-aliasing
///
/// @uptrace{SWS_CORE_10100}
/// @uptrace{SWS_CORE_10101}
/// @uptrace{SWS_CORE_10105}
class Byte final {
    /// @uptrace{SWS_CORE_10102}
    unsigned char mValue;

   public:
    /// @brief Default constructor
    ///
    /// This constructor deliberately does not initialize anything, so that
    /// default-initialization leaves the instance with indeterminate value.
    /// It also deliberately is "defaulted on first declaration", so that this
    /// class does NOT have a "user-provided" default constructor, which is
    /// necessary for achieving the desired equivalence to std::byte.
    ///
    /// @uptrace{SWS_CORE_10104}
    /**
     * @brief Default constructor
     */
    Byte() noexcept = default;

    /// @brief Converting constructor from 'unsigned char'
    ///
    /// @uptrace{SWS_CORE_10103}
    /// @uptrace{SWS_CORE_10106}
    /**
     * @brief Converting constructor from 'unsigned char'
     *
     */
    constexpr explicit Byte( unsigned char value ) noexcept : mValue( value ) {}

    // The compiler-generated special member functions are fine
    // and would be equivalent to these:
    //
    // constexpr byte(byte const& other) noexcept = default;
    // constexpr byte(byte&& other) noexcept = default;
    // constexpr byte& operator=(byte const& other) noexcept = default;
    // constexpr byte& operator=(byte&& other) noexcept = default;
    // ~byte() noexcept = default;

    /// @brief Conversion operator to 'unsigned char'
    ///
    /// @uptrace{SWS_CORE_10107}
    /// @uptrace{SWS_CORE_10108}
    /**
     * @brief Conversion operator to 'unsigned char'
     *
     * @return unsigned char the return value
     */
    constexpr explicit operator unsigned char() const noexcept { return mValue; }
} APD_PACKED;

/// @uptrace{SWS_CORE_10109}
/**
 * @brief Equality comparison for byte ara::core::Byte
 *
 * @param b1 value want to equal
 * @param b2 value want to equal
 * @return true if b1 equal b2
 * @return false otherwise
 */
constexpr bool operator==( Byte b1, Byte b2 ) noexcept {
    return ( static_cast<unsigned char>( b1 ) == static_cast<unsigned char>( b2 ) );
}

/// @uptrace{SWS_CORE_10110}
/**
 * @brief Non-equality comparison for byte ara::core::Byte
 *
 * @param b1 value want to equal
 * @param b2 value want to equal
 * @return true if b1 is not equal b2
 * @return false otherwise
 */
constexpr bool operator!=( Byte b1, Byte b2 ) noexcept {
    return ( static_cast<unsigned char>( b1 ) != static_cast<unsigned char>( b2 ) );
}

}  // namespace uti_internal

#if __cplusplus < 201703L
/// @uptrace{SWS_CORE_04200}
using Byte = uti_internal::Byte;
#else
/// @uptrace{SWS_CORE_04200}
using Byte = std::byte;
#endif

// -------------------------------------------------------------------------

#if ( __cpp_lib_nonmember_container_access - 0 ) >= 201411
using std::data;
using std::empty;
using std::size;
#else
// pre C++17

/// @brief Return a pointer to the block of memory that contains the elements of
/// a container.
/// @tparam Container  a type with a data() method
/// @param c  an instance of Container
/// @returns a pointer to the first element of the container

/// @uptrace{SWS_CORE_04011}
struct in_place_t {
    /// @uptrace{SWS_CORE_04012}
    explicit in_place_t() = default;
};

/// @uptrace{SWS_CORE_04013}
constexpr in_place_t in_place;

/// @uptrace{SWS_CORE_04021}
template <typename T>
struct in_place_type_t {
    /// @uptrace{SWS_CORE_04022}
    explicit in_place_type_t() = default;
};

/// @uptrace{SWS_CORE_04031}
template <size_t I>
struct in_place_index_t {
    /// @uptrace{SWS_CORE_04032}
    /**
     * @brief Default constructor.
     */
    explicit in_place_index_t() = default;
};

/// @uptrace{SWS_CORE_04110}
/**
 * @brief Return a pointer to the block of memory
 *        that contains the elements of a container.
 *
 * @tparam Container a type with a data() method
 * @param c an instance of Container
 * @return decltype( c.data() )  a pointer to the first element of the container
 */
template <typename Container>
constexpr auto data( Container &c ) -> decltype( c.data() ) {
    return c.data();
}

/// @uptrace{SWS_CORE_04111}
/**
 * @brief Return a const_pointer to the block of memory
 *        that contains the elements of a container.
 *
 * @tparam Container a type with a data() method
 * @param c an instance of Container
 * @return decltype( c.data() ) a pointer to the first element of the container
 */
template <typename Container>
constexpr auto data( Container const &c ) -> decltype( c.data() ) {
    return c.data();
}

/// @uptrace{SWS_CORE_04112}
/**
 * @brief Return a pointer to the block of memory
 *        that contains the elements of a raw array.
 * @tparam T  the type of array elements
 * @tparam N  the number of elements in the array
 * @param array  reference to a raw array
 * @returns a pointer to the first element of the array
 */
template <typename T, std::size_t N>
constexpr T *data( T ( &array )[ N ] ) noexcept {
    return array;
}

/// @uptrace{SWS_CORE_04113}
/**
 * @brief Return a pointer to the block of memory
 *        that contains the elements of a std::initializer_list.
 * @tparam E  the type of elements in the std::initializer_list
 * @param il  the std::initializer_list
 * @returns a pointer to the first element of the std::initializer_list
 */
template <typename E>
constexpr E const *data( std::initializer_list<E> il ) noexcept {
    return il.begin();
}

/// @uptrace{SWS_CORE_04120}
/**
 * @brief Return the size of a container.
 * @tparam Container  a type with a data() method
 * @param c  an instance of Container
 * @returns the size of the container
 */
template <typename Container>
constexpr auto size( Container const &c ) -> decltype( c.size() ) {
    return c.size();
}

/// @uptrace{SWS_CORE_04121}
/**
 * @brief Return the size of a raw array.
 * @tparam T  the type of array elements
 * @tparam N  the number of elements in the array
 * @param array  reference to a raw array
 * @returns the size of the array, i.e. N
 */

template <typename T, std::size_t N>
constexpr std::size_t size( T const ( &array )[ N ] ) noexcept {
    return N;
}

/// @uptrace{SWS_CORE_04130}
/**
 * @brief Return whether the given container is empty.
 *
 * @tparam Container a type with a empty() method
 * @param c an instance of Container
 * @return decltype( c.empty() ) true if the container is empty, false otherwise
 */
template <typename Container>
constexpr auto empty( Container const &c ) -> decltype( c.empty() ) {
    return c.empty();
}

/// @uptrace{SWS_CORE_04131}
/**
 * @brief Return whether the given raw array is empty.
 *
 * @tparam T the type of array elements
 * @tparam N the number of elements in the array
 * @return true is empty
 * @return false otherwise
 */
template <typename T, std::size_t N>
constexpr bool empty( T const ( &array )[ N ] ) noexcept {
    return false;
}

/// @uptrace{SWS_CORE_04132}
/**
 * @brief Return whether the given std::initializer_list is empty.
 *
 * @tparam E the type of elements in the std::initializer_list
 * @param il the std::initializer_list
 * @return true if the std::initializer_list is empty
 * @return false otherwise
 */
template <typename E>
constexpr bool empty( std::initializer_list<E> il ) noexcept {
    return il.size() == 0;
}
#endif
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_UTILITY_H_
/* EOF */
