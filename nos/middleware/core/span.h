#ifndef NETAOS_CORE_RESPAN_H
#define NETAOS_CORE_RESPAN_H
#include <limits>

#include "core/array.h"
#include "core/utility.h"

namespace hozon {

namespace netaos {
namespace core {
template <typename T, std::size_t Extent>
class Span;
template <typename T>
struct Is_Ara_Array {
    static bool const value = false;
};

template <typename T, std::size_t N>
struct Is_Ara_Array<hozon::netaos::core::Array<T, N>> {
    static bool const value = true;
};

template <typename T>
struct Is_Ara_Span {
    static bool const value = false;
};

template <typename T, std::size_t Extent>
struct Is_Ara_Span<hozon::netaos::core::Span<T, Extent>> {
    static bool const value = true;
};

void InvalidSpanLengthPrint();
/**
 * @brief A constant for creating Spans with dynamic sizes [SWS_CORE_01901].
 *
 */
constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

/**
 * @brief A view over a contiguous sequence of objects [SWS_CORE_01900].
 *
 * @tpnetaosm T                The type of elements in the Span
 * @tpnetaosm dynamic_extent   The extent to use for this Span
 */
template <typename T, std::size_t Extent = dynamic_extent>
class Span {
   public:
    /**
     * @brief Alias for the type of elements in this Span [SWS_CORE_01911].
     *
     */
    using element_type = T;

    /**
     * @brief Alias for the type of values in this Span [SWS_CORE_01912].
     *
     */
    using value_type = typename std::remove_cv<element_type>::type;

    /**
     * @brief Alias for the type of pnetaosmeters that indicate an index into the Span [SWS_CORE_01913].
     *
     */
    using index_type = std::size_t;

    /**
     * @brief Alias for the type of pnetaosmeters that indicate a difference of indexes into the Span [SWS_CORE_01914].
     *
     */
    using difference_type = std::ptrdiff_t;

    /**
     * @brief This is an AUTOSAR addition that is not contained in std::span [SWS_CORE_01921].
     *
     */
    using size_type = index_type;

    /**
     * @brief Alias type for a pointer to an element [SWS_CORE_01915].
     *
     */
    using pointer = element_type*;

    /**
     * @brief Alias type for a reference to an element [SWS_CORE_01916].
     *
     */
    using reference = element_type&;

    /**
     * @brief The type of an iterator to elements [SWS_CORE_01917].
     *
     */
    using iterator = element_type*;

    /**
     * @brief The type of a const_iterator to elements [SWS_CORE_01918].
     *
     */
    using const_iterator = element_type const*;

    /**
     * @brief The type of a reverse_iterator to elements [SWS_CORE_01919].
     *
     */
    using reverse_iterator = std::reverse_iterator<iterator>;

    /**
     * @brief The type of a const_reverse_iterator to elements [SWS_CORE_01920].
     *
     */
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    template <typename M>
    using type_of_data = decltype(hozon::netaos::core::data(std::declval<M>()));

    /**
     * @brief A constant reflecting the configured Extent of this Span [SWS_CORE_01931].
     *
     */
    static constexpr index_type extent = Extent;

    /**
     * @brief Default constructor [SWS_CORE_01941].
     *
     */
    constexpr Span() noexcept {}

    /**
     * @brief Construct a new Span from the given pointer and size [SWS_CORE_01942].
     *
     * @pnetaosm[in] ptr     the pointer
     * @pnetaosm[in] count   the number of elements to take from ptr
     */
    constexpr Span(pointer ptr, index_type count) noexcept : spanData_(ptr), totalLength_(count) {}

    /**
     * @brief Construct a new Span from the open range between [firstElem, lastElem) [SWS_CORE_01943].
     *
     * @pnetaosm[in] firstElem   pointer to the first element
     * @pnetaosm[in] lastElem    pointer to past the last element
     */
    constexpr Span(pointer firstElem, pointer lastElem) {
        if (lastElem < firstElem || (Extent != dynamic_extent && lastElem - firstElem != Extent)) {
            InvalidSpanLengthPrint();
        } else {
            spanData_ = firstElem;
            totalLength_ = lastElem - firstElem;
        }
    }

    /**
     * @brief Construct a new Span from the given raw array.
     *
     * @tpnetaosm    N     the size of the raw array
     * @pnetaosm[in] arr   the array
     */
    template <std::size_t N, typename std::enable_if<std::is_convertible<typename std::remove_pointer<type_of_data<element_type (&)[N]>>::type (*)[], element_type (*)[]>::value &&
                                                     (Extent == dynamic_extent || N == Extent)>::type* = nullptr>
    constexpr Span(element_type (&arr)[N]) noexcept : spanData_(&arr[0]), totalLength_(N) {}

    /**
     * @brief Construct a new Span from the given Array [SWS_CORE_01945].
     *
     * @tpnetaosm N         the size of the array
     * @pnetaosm[in] arr    the array
     */
    template <std::size_t N, typename std::enable_if<
                                 std::is_convertible<typename std::remove_pointer<decltype(std::declval<hozon::netaos::core::Array<value_type, N>>().data())>::type (*)[], element_type (*)[]>::value &&
                                 (Extent == dynamic_extent || N == Extent)>::type* = nullptr>
    constexpr Span(hozon::netaos::core::Array<value_type, N>& arr) noexcept : spanData_(arr.data()), totalLength_(N) {}

    /**
     * @brief Construct a new Span from the given const Array [SWS_CORE_01946].
     *
     * @tpnetaosm    N       the size of the Array
     * @pnetaosm[in] arr     the array
     */
    template <std::size_t N, typename std::enable_if<
                                 std::is_convertible<typename std::remove_pointer<decltype(std::declval<hozon::netaos::core::Array<value_type, N>>().data())>::type (*)[], element_type (*)[]>::value &&
                                 (Extent == dynamic_extent || N == Extent)>::type* = nullptr>
    constexpr Span(hozon::netaos::core::Array<value_type, N> const& arr) noexcept : spanData_(arr.data()), totalLength_(N) {}

    /**
     * @brief Construct a new Span from the given container [SWS_CORE_01947].
     *
     * @tpnetaosm    Container   the type of container
     * @pnetaosm[in] cont        the container
     */
    template <typename Container, typename std::enable_if<!std::is_array<typename std::remove_cv<Container>::type>::value && !Is_Ara_Array<typename std::remove_cv<Container>::type>::value &&
                                                          !Is_Ara_Span<typename std::remove_cv<Container>::type>::value>::type* = nullptr>
    constexpr Span(Container& cont) {
        if (Extent != dynamic_extent && hozon::netaos::core::size(cont) != Extent) {
            InvalidSpanLengthPrint();
        } else {
            spanData_ = hozon::netaos::core::data(cont);
            totalLength_ = hozon::netaos::core::size(cont);
        }
    }

    /**
     * @brief Construct a new Span from the given const container [SWS_CORE_01948].
     *
     * @tpnetaosm     Container    the type of container
     * @pnetaosm[in] cont          the container
     */
    template <typename Container, typename std::enable_if<!std::is_array<typename std::remove_cv<Container>::type>::value && !Is_Ara_Array<typename std::remove_cv<Container>::type>::value &&
                                                          !Is_Ara_Span<typename std::remove_cv<Container>::type>::value>::type* = nullptr>
    constexpr Span(Container const& cont) {
        if (Extent != dynamic_extent && hozon::netaos::core::size(cont) != Extent) {
            InvalidSpanLengthPrint();
        } else {
            spanData_ = hozon::netaos::core::data(cont);
            totalLength_ = hozon::netaos::core::size(cont);
        }
    }

    /**
     * @brief Converting constructor [SWS_CORE_01950].
     *
     * @tpnetaosm    U     the type of elements within the other Span
     * @tpnetaosm    N     the Extent of the other Span
     * @pnetaosm[in] s     the other Span instance
     */
    template <typename U, std::size_t N, typename = typename std::enable_if<(Extent == dynamic_extent || Extent == N) && std::is_convertible<U (*)[], element_type (*)[]>::value>::type*>
    constexpr Span(Span<U, N> const& s) noexcept : spanData_(s.data()), totalLength_(s.size()) {}

    /**
     * @brief Destroy the Span object [SWS_CORE_01951].
     *
     */
    ~Span() noexcept = default;

    /**
     * @brief Copy construct a new Span from another instance [SWS_CORE_01949].
     *
     */
    constexpr Span(Span const& other) noexcept = default;

    /**
     * @brief Copy construct a new Span from another instance [SWS_CORE_01949].
     *
     */
    Span& operator=(Span const& other) noexcept = default;

    /**
     * @brief Return a subspan containing only the first elements of this Span [SWS_CORE_01961].
     *
     * @tpnetaosm Count[in]     the number of elements to take over
     * @return constexpr Span<element_type, Count>  the subspan
     */
    template <std::size_t Count>
    constexpr Span<element_type, Count> first() const {
        if (Count > totalLength_) {
            InvalidSpanLengthPrint();
            return Span<element_type, Count>();
        } else {
            return Span<element_type, Count>(spanData_, Count);
        }
    }

    /**
     * @brief Return a subspan containing only the first elements of this Span [SWS_CORE_01962].
     *
     * @pnetaosm count[in]     the number of elements to take over
     * @return constexpr Span<element_type, dynamic_extent>   the subspan
     */
    constexpr Span<element_type, dynamic_extent> first(index_type count) const {
        if (count > totalLength_) {
            InvalidSpanLengthPrint();
            return Span<element_type, dynamic_extent>();
        } else {
            return Span<element_type, dynamic_extent>(spanData_, count);
        }
    }

    /**
     * @brief Return a subspan containing only the last elements of this Span [SWS_CORE_01963].
     *
     * @tpnetaosm Count the number of elements to take over
     * @return constexpr Span<element_type, Count> the subspan
     */
    template <std::size_t Count>
    constexpr Span<element_type, Count> last() const {
        if (Count > totalLength_) {
            InvalidSpanLengthPrint();
            return Span<element_type, Count>();
        } else {
            return Span<element_type, Count>(spanData_ + totalLength_ - Count, Count);
        }
    }

    /**
     * @brief Return a subspan containing only the last elements of this Span [SWS_CORE_01964].
     *
     * @pnetaosm[in]   count    the number of elements to take over
     * @return constexpr Span<element_type, dynamic_extent>   the subspan
     */
    constexpr Span<element_type, dynamic_extent> last(index_type count) const {
        if (count > totalLength_) {
            InvalidSpanLengthPrint();
            return Span<element_type, dynamic_extent>();
        } else {
            return Span<element_type, dynamic_extent>(spanData_ + totalLength_ - count, count);
        }
    }

    /**
     * @brief Return a subspan of this Span [SWS_CORE_01965].
     *
     * @tpnetaosm Offset           offset into this Span from which to start
     * @tpnetaosm dynamic_extent   the number of elements to take over
     * @return constexpr auto   the subspan
     */
    template <std::size_t Offset, std::size_t Count = dynamic_extent>
    constexpr auto subspan() const {
        if ((Count == dynamic_extent && totalLength_ < Offset) ||
            // Divided by 2.0 to avoid potential integer overflow
            (Count != dynamic_extent && (totalLength_ / 2.0) < (Offset / 2.0) + (Count / 2.0))) {
            InvalidSpanLengthPrint();
            return Span();
        } else {
            return Span(spanData_ + Offset, Count != dynamic_extent ? Count : totalLength_ - Offset);
        }
    }

    /**
     * @brief Return a subspan of this Span [SWS_CORE_01966].
     *
     * @pnetaosm offset[in]   offset into this Span from which to start
     * @pnetaosm count[in]    the number of elements to take over
     * @return constexpr Span<element_type, dynamic_extent>   the subspan
     */
    constexpr Span<element_type, dynamic_extent> subspan(index_type offset, index_type count = dynamic_extent) {
        if ((count == dynamic_extent && totalLength_ < offset) ||
            // Divided by 2.0 to avoid potential integer overflow
            (count != dynamic_extent && (totalLength_ / 2.0) < (offset / 2.0) + (count / 2.0))) {
            InvalidSpanLengthPrint();
            return Span();
        } else {
            return Span(spanData_ + offset, count != dynamic_extent ? count : totalLength_ - offset);
        }
    }

    /**
     * @brief Return the size of this Span [SWS_CORE_01967].
     *
     * @return index_type the number of elements contained in this Span
     */
    constexpr index_type size() const noexcept { return totalLength_; }

    /**
     * @brief Return the size of this Span in bytes [SWS_CORE_01968].
     *
     * @return index_type   the number of bytes covered by this Span
     */
    constexpr index_type size_bytes() const noexcept { return totalLength_ * sizeof(element_type); }

    /**
     * @brief Return whether this Span is empty [SWS_CORE_01969].
     *
     * @return bool   true if this Span contains 0 elements, false otherwise
     */
    constexpr bool empty() const noexcept { return totalLength_ == 0 ? true : false; }

    /**
     * @brief Return a reference to the n-th element of this Span [SWS_CORE_01970].
     *
     * @pnetaosm idx the index into this Span
     * @return reference  the reference
     */
    constexpr reference operator[](index_type idx) const { return *(spanData_ + idx); }

    /**
     * @brief Return a pointer to the start of the memory block covered by this Span [SWS_CORE_01971].
     *
     * @return constexpr pointer
     */
    constexpr pointer data() const noexcept { return spanData_; }

    /**
     * @brief Return an iterator pointing to the first element of this Span [SWS_CORE_01972].
     *
     * @return constexpr iterator
     */
    constexpr iterator begin() const noexcept { return spanData_; }

    /**
     * @brief Return an iterator pointing past the last element of this Span [SWS_CORE_01973].
     *
     * @return constexpr iterator
     */
    constexpr iterator end() const noexcept { return spanData_ + totalLength_; }

    /**
     * @brief Return a const_iterator pointing to the first element of this Span [SWS_CORE_01974].
     *
     * @return constexpr const_iterator
     */
    constexpr const_iterator cbegin() const noexcept { return spanData_; }

    /**
     * @brief Return a const_iterator pointing past the last element of this Span [SWS_CORE_01975].
     *
     * @return constexpr const_iterator
     */
    constexpr const_iterator cend() const noexcept { return spanData_ + totalLength_; }

    /**
     * @brief Return a reverse_iterator pointing to the last element of this Span [SWS_CORE_01976].
     *
     * @return constexpr reverse_iterator
     */
    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }

    /**
     * @brief Return a reverse_iterator pointing past the first element of this Span [SWS_CORE_01977].
     *
     * @return constexpr reverse_iterator
     */
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

    /**
     * @brief Return a const_reverse_iterator pointing to the last element of this Span [SWS_CORE_01978].
     *
     * @return constexpr const_reverse_iterator
     */
    constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    /**
     * @brief Return a const_reverse_iterator pointing past the first element of this Span[SWS_CORE_01979].
     *
     * @return constexpr const_reverse_iterator
     */
    constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

   private:
    pointer spanData_{nullptr};
    index_type totalLength_{0U};
};

/**
 * @brief Create a new Span from the given pointer and size [SWS_CORE_01990].
 *
 * @tpnetaosm    T       the type of elements
 * @pnetaosm[in] ptr     the pointer
 * @pnetaosm[in] count   the number of elements to take from ptr
 * @return Span<T>    the new Span
 */
template <typename T>
constexpr Span<T> MakeSpan(T* ptr, typename Span<T>::index_type count) {
    return Span<T>(ptr, count);
}

/**
 * @brief Create a new Span from the open range between [firstElem, lastElem) [SWS_CORE_01991].
 *
 * @tpnetaosm    T             the type of elements
 * @pnetaosm[in] firstElem     pointer to the first element
 * @pnetaosm[in] lastElem      pointer to past the last element
 * @return    Span<T>       the new Span
 */
template <typename T>
constexpr Span<T> MakeSpan(T* firstElem, T* lastElem) {
    return Span<T>(firstElem, lastElem);
}

/**
 * @brief Create a new Span from the given raw array [SWS_CORE_01992].
 *
 * @tpnetaosm    T            the type of elements
 * @tpnetaosm    N            the size of the raw array
 * @pnetaosm[in] arr          the raw array
 * @return    Span<T, N>   the new Span
 */
template <typename T, std::size_t N>
constexpr Span<T, N> MakeSpan(T (&arr)[N]) noexcept {
    return Span<T, N>(arr);
}

/**
 * @brief Create a new Span from the given container [SWS_CORE_01993].
 *
 * @tpnetaosm    Container       the type of container
 * @pnetaosm[in] cont            the container
 * @return   Span<typename Container::value_type>  the new Span
 */
template <typename Container>
constexpr Span<typename Container::value_type> MakeSpan(Container& cont) {
    return Span<typename Container::value_type>(cont);
}

/**
 * @brief Create a new Span from the given const container [SWS_CORE_01994].
 *
 * @tpnetaosm     Container   the type of container
 * @pnetaosm[in]  cont        the container
 * @return     Span<typename Container::value_type const>  the new Span
 */
template <typename Container>
constexpr Span<typename Container::value_type const> MakeSpan(Container const& cont) {
    return Span<typename Container::value_type const>(cont);
}
}  // namespace core
}  // namespace netaos
}  // namespace hozon
#endif
