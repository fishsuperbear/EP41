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
 * @file span.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_SPAN_H_
#define APD_ARA_CORE_SPAN_H_

#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <type_traits>

#include "ara/core/array.h"
#include "ara/core/utility.h"

namespace ara {
namespace core {
inline namespace _19_11 {

// @uptrace{SWS_CORE_01901}
constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

template <typename T, std::size_t Extent>
class Span;

namespace internal {

/**
 * @brief span checker
 *
 * @tparam T the type of elements in the Span
 */
template <typename T>
struct is_ara_core_span_checker : public std::false_type {};

/**
 * @brief span checker
 *
 * @tparam T the type of elements in the Span
 * @tparam Extent the extent to use for this Span
 */
template <typename T, std::ptrdiff_t Extent>
struct is_ara_core_span_checker<core::Span<T, Extent>> : public std::true_type {};

/**
 * @brief judge is span
 *
 * @tparam T the type of elements in the Span
 */
template <typename T>
struct is_ara_core_span : public is_ara_core_span_checker<typename std::remove_cv<T>::type> {};

/**
 * @brief array checker
 *
 * @tparam T the type of elements in the Span
 */
template <typename T>
struct is_ara_core_array_checker : public std::false_type {};

/**
 * @brief array checker
 *
 * @tparam T the type of elements in the Span
 * @tparam Extent the extent to use for this Span
 */
template <typename T, std::size_t N>
struct is_ara_core_array_checker<core::Array<T, N>> : public std::true_type {};

/**
 * @brief judge is array
 *
 * @tparam T the type of elements in the Span
 */
template <typename T>
struct is_ara_core_array : public is_ara_core_array_checker<typename std::remove_cv<T>::type> {};

/**
 * @brief judge is complete
 *
 * @tparam ypename any type
 * @tparam typename the type of size_t
 */
template <typename, typename = size_t>
struct is_complete : std::false_type {};

/**
 * @brief judge is complete
 *
 * @tparam T the type of elements in the Span
 */
template <typename T>
struct is_complete<T, decltype( sizeof( T ) )> : std::true_type {};

}  // namespace internal

/// @brief A view over a contiguous sequence of objects

/**
 * @class Span
 *
 * @brief Class for Span @uptrace{SWS_CORE_01900}
 */
template <typename T, std::size_t Extent = dynamic_extent>
class Span {
   public:
    template <typename U>
    using type_of_data = decltype( ara::core::data( std::declval<U>() ) );

    /// @uptrace{SWS_CORE_01911}
    using element_type = T;
    /// @uptrace{SWS_CORE_01912}
    using value_type = typename std::remove_cv<element_type>::type;
    /// @uptrace{SWS_CORE_01913}
    using index_type = std::size_t;
    /// @uptrace{SWS_CORE_01914}
    using difference_type = std::ptrdiff_t;
    /// @uptrace{SWS_CORE_01915}
    using pointer = element_type *;
    /// @uptrace{SWS_CORE_01916}
    using reference = element_type &;
    /// @uptrace{SWS_CORE_01917}
    using iterator = element_type *;
    /// @uptrace{SWS_CORE_01918}
    using const_iterator = element_type const *;
    /// @uptrace{SWS_CORE_01919}
    using reverse_iterator = std::reverse_iterator<iterator>;
    /// @uptrace{SWS_CORE_01920}
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // Not in C++ standard, but we need it for GMock's IsSize()
    /// @uptrace{SWS_CORE_01921}
    using size_type = index_type;

    /// @uptrace{SWS_CORE_01931}
    static constexpr index_type extent = Extent;

    // 21.7.3.2, constructors, copy, and assignment

    // @uptrace{SWS_CORE_01941}
    /**
     * @brief Default constructor.
     *        This constructor shall not participate in overload resolution
     *        unless Extent <= 0 is true.
     */
    template <typename U = void,
              typename = typename std::enable_if<Extent == dynamic_extent || Extent == 0, U>::type>
    constexpr Span() noexcept : mData( nullptr ), mExtent( 0 ) {}

    /// @uptrace{SWS_CORE_01942}
    /**
     * @brief Construct a new Span from the given pointer and size.
     *        [ptr, ptr + count) shall be a valid range.
     *        If Extent is not equal to dynamic_extent, then count shall be equal to Extent
     *
     * @param ptr the pointer
     * @param count the number of elements to take from ptr
     */
    constexpr Span( pointer ptr, index_type count ) : mData( ptr ), mExtent( count ) {
        if ( Extent != dynamic_extent && count != Extent ) {
            assert( !"inconsistent extent" );
        }
    }

    /// @uptrace{SWS_CORE_01943}
    /**
     * @brief Construct a new Span from the open range between [firstElem, lastElem).
     *        [first, last) shall be a valid range.
     *        If @ extent is not equal to dynamic_extent,
     *        then (last - first) shall be equal to Extent.
     *
     * @param firstElem pointer to the first element
     * @param lastElem pointer to past the last element
     */
    constexpr Span( pointer firstElem, pointer lastElem )
        : Span( firstElem, lastElem - firstElem ) {
        assert( std::distance( firstElem, lastElem ) >= 0 );
    }

    /// @uptrace{SWS_CORE_01944}
    /**
     * @brief Construct a new Span from the given raw array.
     *        This constructor shall not participate in overload resolution unless:
     *        Extent == dynamic_extent || N == Extent is true,
     *        and std::remove_pointer<decltype(ara::core::data(arr))>::type(*)[] is
     *        convertible to T(*)[].
     *
     * @tparam N the size of the raw array
     * @tparam std::enable_if<
     * ( Extent == dynamic_extent || Extent == N ) &&
     * std::is_convertible<typename std::remove_pointer<type_of_data<element_type ( & )[ N ]>>::type
     * ( * )[], element_type ( * )[]>::value>::type  the condition
     *
     * @param arr the raw array
     */
    template <std::size_t N, typename = typename std::enable_if<
                                 ( Extent == dynamic_extent || Extent == N ) &&
                                 std::is_convertible<typename std::remove_pointer<type_of_data<
                                                         element_type ( & )[ N ]>>::type ( * )[],
                                                     element_type ( * )[]>::value>::type>
    constexpr Span( element_type ( &arr )[ N ] ) noexcept : Span( std::addressof( arr[ 0 ] ), N ) {}

    /// @uptrace{SWS_CORE_01945}
    /**
     * @brief Construct a new Span from the given Array.
     *        This constructor shall not participate in overload resolution unless:
     *        Extent == dynamic_extent || N == Extent is true,
     *        and std::remove_pointer<decltype(ara::core::data(arr))>::type(*)[]
     *        is convertible to T(*)[].
     *
     * @tparam N the size of the Array
     * @tparam std::enable_if<
     * ( Extent == dynamic_extent || Extent == N ) &&
     * std::is_convertible<typename std::remove_pointer<decltype(
     * std::declval<Array<value_type, N>>().data() )>::type ( * )[],
     * element_type ( * )[]>::value>::type
     *
     * @param arr the raw array
     */
    template <
        std::size_t N,
        typename = typename std::enable_if<
            ( Extent == dynamic_extent || Extent == N ) &&
            std::is_convertible<typename std::remove_pointer<decltype(
                                    std::declval<Array<value_type, N>>().data() )>::type ( * )[],
                                element_type ( * )[]>::value>::type>
    constexpr Span( Array<value_type, N> &arr ) noexcept : Span( arr.data(), N ) {}

    /// @uptrace{SWS_CORE_01946}
    /**
     * @brief Construct a new Span from the given const Array.
     *        This constructor shall not participate in overload resolution unless:
     *        Extent == dynamic_extent || N == Extent is true,
     *        and std::remove_pointer<decltype(ara::core::data(arr))>::type(*)[]
     *        is convertible to T(*)[].
     *
     * @tparam N the size of the Array
     * @tparam std::enable_if<
     * ( Extent == dynamic_extent || Extent == N ) &&
     * std::is_convertible<
     * typename std::remove_pointer<decltype(
     * std::declval<Array<value_type, N> const>().data() )>::type ( * )[],
     * element_type ( * )[]>::value>::type
     *
     * @param arr the raw array
     */
    template <std::size_t N,
              typename = typename std::enable_if<
                  ( Extent == dynamic_extent || Extent == N ) &&
                  std::is_convertible<
                      typename std::remove_pointer<decltype(
                          std::declval<Array<value_type, N> const>().data() )>::type ( * )[],
                      element_type ( * )[]>::value>::type>
    constexpr Span( Array<value_type, N> const &arr ) noexcept : Span( arr.data(), N ) {}

    /// @uptrace{SWS_CORE_01947}
    /**
     * @brief Construct a new Span from the given container.
     *        [ara::core::data(cont), ara::core::data(cont) + ara::core::size(cont))
     *        shall be a valid range. If Extent is not equal to dynamic_extent,
     *        then ara::core::size(cont) shall be equal to Extent.
     *        These constructors shall not participate in overload resolution unless:
     *        Container is not a specialization of Span,
     *        Container is not a specialization of Array, std::is_array<Container>::value is false,
     *        ara::core::data(cont) and ara::core::size(cont) are both well-formed,
     *        and std::remove_pointer<decltype(ara::core::data(cont))>::type(*)[]
     *        is convertible to T(*)[].
     *
     * @tparam Container the type of container
     * @tparam std::enable_if<!internal::is_ara_core_span<Container>::value &&
     * !internal::is_ara_core_array<Container>::value &&
     * !std::is_array<Container>::value>::type
     *
     * @param cont the container
     */
    template <typename Container,
              typename = typename std::enable_if<!internal::is_ara_core_span<Container>::value &&
                                                 !internal::is_ara_core_array<Container>::value &&
                                                 !std::is_array<Container>::value>::type>
    constexpr Span( Container &cont )
        : Span( core::data( cont ), static_cast<index_type>( core::size( cont ) ) ) {
        assert( std::distance( core::data( cont ), core::data( cont ) + core::size( cont ) ) >= 0 );
    }

    /// @uptrace{SWS_CORE_01948}
    /**
     * @brief Construct a new Span from the given const container.
     *        [ara::core::data(cont), ara::core::data(cont) + ara::core::size(cont))
     *        shall be a valid range. If Extent is not equal to dynamic_extent,
     *        then ara::core::size(cont) shall be equal to Extent.
     *        These constructors shall not participate in overload resolution unless:
     *        Container is not a specialization of Span,
     *        Container is not a specialization of Array,
     *        std::is_array<Container>::value is false,
     *        ara::core::data(cont) and ara::core::size(cont) are both well-formed,
     *        and std::remove_pointer<decltype(ara::core::data(cont))>::type(*)[]
     *        is convertible to T(*)[].
     *
     * @tparam Container the type of container
     * @tparam std::enable_if<!internal::is_ara_core_span<Container>::value &&
     * !internal::is_ara_core_array<Container>::value &&
     * !std::is_array<Container>::value>::type
     *
     * @param cont the container
     */
    template <typename Container,
              typename = typename std::enable_if<!internal::is_ara_core_span<Container>::value &&
                                                 !internal::is_ara_core_array<Container>::value &&
                                                 !std::is_array<Container>::value>::type>
    constexpr Span( Container const &cont )
        : Span( core::data( cont ), static_cast<index_type>( core::size( cont ) ) ) {
        assert( std::distance( core::data( cont ), core::data( cont ) + core::size( cont ) ) >= 0 );
    }

    /// @uptrace{SWS_CORE_01949}
    /**
     * @brief Copy construct a new Span from another instance.
     *
     * @param other the other instance
     */
    constexpr Span( Span const &other ) noexcept = default;

    /// @uptrace{SWS_CORE_01950}
    /**
     * @brief Converting constructor.
     *        This ctor allows construction of a cv-qualified Span from a normal Span,
     *        and also of a dynamic_extent-Span<> from a static extent-one.
     *
     * @tparam U the type of elements within the other Span
     * @tparam N the Extent of the other Span
     * @tparam * the other Span instance
     */
    template <typename U, std::size_t N,
              typename = typename std::enable_if<
                  ( Extent == dynamic_extent || Extent == N ) &&
                  std::is_convertible<U ( * )[], element_type ( * )[]>::value>::type *>
    constexpr Span( Span<U, N> const &s ) noexcept : Span( s.data(), s.size() ) {}

    /// @uptrace{SWS_CORE_01951}
    /**
     * @brief Destructor.
     *
     */
    ~Span() noexcept = default;

    // Not "constexpr" because that would make it also "const" on C++11 compilers.
    /// @uptrace{SWS_CORE_01952}
    /**
     * @brief This operator is not constexpr because that would make it implicitly const in C++11.
     *
     * @param other the other instance
     * @return Span& *this
     */
    Span &operator=( Span const &other ) noexcept = default;

    // 21.7.3.3, subviews

    /// @uptrace{SWS_CORE_01961}
    /**
     * @brief Return a subspan containing only the first elements of this Span.
     *
     * @tparam Count the number of elements to take over
     * @return constexpr Span<element_type, Count> the subspan
     */
    template <std::size_t Count>
    constexpr Span<element_type, Count> first() const {
        static_assert( 0 <= Count, "subview size cannot be negative" );
        assert( Count <= size() );

        return { mData, Count };
    }

    /// @uptrace{SWS_CORE_01962}
    /**
     * @brief Return a subspan containing only the first elements of this Span.
     *
     * @param count the number of elements to take over
     * @return constexpr Span<element_type, dynamic_extent> the subspan
     */
    constexpr Span<element_type, dynamic_extent> first( index_type count ) const {
        assert( 0 <= count && count <= size() );

        return { mData, count };
    }

    /// @uptrace{SWS_CORE_01963}
    /**
     * @brief Return a subspan containing only the last elements of this Span.
     *
     * @tparam Count the number of elements to take over the subspan
     * @return constexpr Span<element_type, Count>
     */
    template <std::size_t Count>
    constexpr Span<element_type, Count> last() const {
        static_assert( 0 <= Count, "subview size cannot be negative" );
        assert( Count <= size() );

        return { mData + ( size() - Count ), Count };
    }

    /// @uptrace{SWS_CORE_01964}
    /**
     * @brief Return a subspan containing only the last elements of this Span.
     *
     * @param count the number of elements to take over
     * @return constexpr Span<element_type, dynamic_extent> the subspan
     */
    constexpr Span<element_type, dynamic_extent> last( index_type count ) const {
        assert( 0 <= count && count <= size() );

        return { mData + ( size() - count ), count };
    }

    // need proper C++11 compatible return type
    /// @uptrace{SWS_CORE_01965}
    /**
     * @brief Return a subspan of this Span.
     *        The second template argument of the returned Span type is:
     *        Count != dynamic_extent ?
     *        Count : (Extent != dynamic_extent ? Extent - Offset : dynamic_extent)
     *
     * @tparam Offset offset into this Span from which to start
     * @tparam Count the number of elements to take over
     *
     * @return Span< element_type, SEE_BELOW > the subspan
     */
    template <std::size_t Offset, std::size_t Count = dynamic_extent>
    constexpr auto subspan() const
        -> Span<element_type,
                Count != dynamic_extent
                    ? Count
                    : ( Extent != dynamic_extent ? Extent - Offset : dynamic_extent )> {
        assert( 0 <= Offset && Offset <= size() );
        assert( Count == dynamic_extent || ( Count >= 0 && Offset + Count <= size() ) );

        constexpr index_type E =
            ( Count != dynamic_extent )
                ? Count
                : ( Extent != dynamic_extent ? Extent - Offset : dynamic_extent );

        return Span<element_type, E>( mData + Offset,
                                      Count != dynamic_extent ? Count : size() - Offset );
    }

    /// @uptrace{SWS_CORE_01966}
    /**
     * @brief Return a subspan of this Span.
     *
     * @param offset [IN] offset into this Span from which to start
     * @param count [IN] the number of elements to take over
     * @return constexpr Span<element_type, dynamic_extent> the subspan
     */
    constexpr Span<element_type, dynamic_extent> subspan(
        index_type offset, index_type count = dynamic_extent ) const {
        assert( 0 <= offset && offset <= size() );
        assert( count == dynamic_extent || ( count >= 0 && offset + count <= size() ) );

        return { mData + offset, count == dynamic_extent ? size() - offset : count };
    }

    // 21.7.3.4, observers

    /// @uptrace{SWS_CORE_01967}
    /**
     * @brief Return the size of this Span.
     *
     * @return constexpr index_type the number of elements contained in this Span
     */
    constexpr index_type size() const noexcept { return mExtent; }

    /// @uptrace{SWS_CORE_01968}
    /**
     * @brief Return the size of this Span in bytes.
     *
     * @return constexpr index_type the number of bytes covered by this Span
     */
    constexpr index_type size_bytes() const noexcept { return mExtent * sizeof( T ); }

    /// @uptrace{SWS_CORE_01969}
    /**
     * @brief Return whether this Span is empty.
     *
     * @return true if this Span contains 0 elements
     * @return false otherwise
     */
    constexpr bool empty() const noexcept { return mExtent == 0; }

    // 21.7.3.5, element access

    /// @uptrace{SWS_CORE_01970}
    /**
     * @brief Return a reference to the n-th element of this Span.
     *
     * @param idx the index into this Span
     * @return constexpr reference the reference
     */
    constexpr reference operator[]( index_type idx ) const { return mData[ idx ]; }

    /// @uptrace{SWS_CORE_01971}
    /**
     * @brief Return a pointer to the start of the memory block covered by this Span.
     *
     * @return constexpr pointer the pointer
     */
    constexpr pointer data() const noexcept { return mData; }

    // 21.7.3.6, iterator support

    /// @uptrace{SWS_CORE_01972}
    /**
     * @brief Return an iterator pointing to the first element of this Span.
     *
     * @return constexpr iterator the iterator
     */
    constexpr iterator begin() const noexcept { return &mData[ 0 ]; }

    /// @uptrace{SWS_CORE_01973}
    /**
     * @brief Return an iterator pointing past the last element of this Span.
     *
     * @return constexpr iterator the iterator
     */
    constexpr iterator end() const noexcept { return &mData[ mExtent ]; }

    /// @uptrace{SWS_CORE_01974}
    /**
     * @brief Return a const_iterator pointing to the first element of this Span.
     *
     * @return constexpr const_iterator
     */
    constexpr const_iterator cbegin() const noexcept { return &mData[ 0 ]; }

    /// @uptrace{SWS_CORE_01975}
    /**
     * @brief Return a const_iterator pointing past the last element of this Span.
     *
     * @return constexpr const_iterator the const_iterator
     */
    constexpr const_iterator cend() const noexcept { return &mData[ mExtent ]; }

    /// @uptrace{SWS_CORE_01976}
    /**
     * @brief Return a reverse_iterator pointing to the last element of this Span.
     *
     * @return constexpr reverse_iterator the reverse_iterator
     */
    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator( end() ); }

    /// @uptrace{SWS_CORE_01977}
    /**
     * @brief Return a reverse_iterator pointing past the first element of this Span.
     *
     * @return constexpr reverse_iterator the reverse_iterator
     */
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator( begin() ); }

    /// @uptrace{SWS_CORE_01978}
    /**
     * @brief Return a const_reverse_iterator pointing to the last element of this Span.
     *
     * @return constexpr const_reverse_iterator the const_reverse_iterator
     */
    constexpr const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator( end() );
    }

    /// @uptrace{SWS_CORE_01979}
    /**
     * @brief Return a const_reverse_iterator pointing past the first element of this Span.
     *
     * @return constexpr const_reverse_iterator the reverse_iterator
     */
    constexpr const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator( begin() );
    }

   private:
    pointer    mData;
    index_type mExtent;

   private:
    static_assert( Extent == dynamic_extent || Extent >= 0, "invalid extent for a Span" );
    static_assert( std::is_object<T>::value, "Span cannot contain reference or void types" );
    static_assert( !std::is_abstract<T>::value, "Span cannot contain abstract types" );
    static_assert( internal::is_complete<T>::value, "Span cannot contain incomplete types" );
};

#if ( __cpp_deduction_guides - 0 ) >= 201703L
template <typename T, std::size_t N>
Span( T ( & )[ N ] ) -> Span<T, N>;

template <typename T, std::size_t N>
Span( Array<T, N> & ) -> Span<T, N>;

template <typename T, std::size_t N>
Span( Array<T, N> const & ) -> Span<T const, N>;

template <typename Container>
Span( Container & ) -> Span<typename Container::value_type>;

template <typename Container>
Span( Container const & ) -> Span<const typename Container::value_type>;
#endif  // C++17 deduction guides

/// @uptrace{SWS_CORE_01990}
/**
 * @brief Create a new Span from the given pointer and size.
 *
 * @tparam T the type of elements
 * @param ptr the pointer
 * @param count the number of elements to take from ptr
 * @return constexpr Span<T> the new Span
 */
template <typename T>
constexpr Span<T> MakeSpan( T *ptr, typename Span<T>::index_type count ) {
    return Span<T>( ptr, count );
}

/// @uptrace{SWS_CORE_01991}
/**
 * @brief Create a new Span from the open range between [firstElem, lastElem).
 *
 * @tparam T the type of elements
 * @param firstElem pointer to the first element
 * @param lastElem pointer to past the last element
 * @return constexpr Span<T> the new Span
 */
template <typename T>
constexpr Span<T> MakeSpan( T *firstElem, T *lastElem ) {
    return Span<T>( firstElem, lastElem );
}

/// @uptrace{SWS_CORE_01992}
/**
 * @brief Create a new Span from the given raw array.
 *
 * @tparam T the type of elements
 * @tparam N the size of the raw array
 * @param arr the raw array
 * @return constexpr Span<T, N> the new Span
 */
template <typename T, std::size_t N>
constexpr Span<T, N> MakeSpan( T ( &arr )[ N ] ) noexcept {
    return Span<T, N>( arr );
}

/// @uptrace{SWS_CORE_01993}
/**
 * @brief Create a new Span from the given container.
 *
 * @tparam Container the type of container
 * @param cont the container
 * @return constexpr Span<typename Container::value_type> the new Span
 */
template <typename Container>
constexpr Span<typename Container::value_type> MakeSpan( Container &cont ) {
    return Span<typename Container::value_type>( cont );
}

/// @uptrace{SWS_CORE_01994}
/**
 * @brief Create a new Span from the given const container.
 *
 * @tparam Container the type of container
 * @param cont the container
 * @return constexpr Span<typename Container::value_type const> the new Span
 */
template <typename Container>
constexpr Span<typename Container::value_type const> MakeSpan( Container const &cont ) {
    return Span<typename Container::value_type const>( cont );
}
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_SPAN_H_
/* EOF */
