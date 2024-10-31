#pragma once

#include <stddef.h>
#include <assert.h>
#include <atomic>

namespace folly {
namespace detail {

inline size_t next_pow_two(size_t v) {
    size_t x = 1;
    while (x < v) {
        x <<= 1;
    }
    return x;
}

template <typename Atomic>
inline bool atomic_fetch_set_default(
    Atomic& atomic,
    std::size_t bit,
    std::memory_order order) {
    
    using Integer = decltype(atomic.load());
    auto mask = Integer{0b1} << static_cast<Integer>(bit);
    return (atomic.fetch_or(mask, order) & mask);
}

template <typename Atomic>
inline bool atomic_fetch_set(Atomic& atomic, std::size_t bit, std::memory_order mo) {
    using Integer = decltype(atomic.load());
    static_assert(std::is_unsigned<Integer>{}, "");
    static_assert(!std::is_const<Atomic>{}, "");
    assert(bit < (sizeof(Integer) * 8));

    // otherwise default to the default implementation using fetch_or()
    return atomic_fetch_set_default(atomic, bit, mo);
}

template <typename Atomic>
bool atomic_fetch_reset_default(
    Atomic& atomic,
    std::size_t bit,
    std::memory_order order) {

    using Integer = decltype(atomic.load());
    auto mask = Integer{0b1} << static_cast<Integer>(bit);
    return (atomic.fetch_and(~mask, order) & mask);
}

template <typename Atomic>
bool atomic_fetch_reset(Atomic& atomic, std::size_t bit, std::memory_order mo) {
    using Integer = decltype(atomic.load());
    static_assert(std::is_unsigned<Integer>{}, "");
    static_assert(!std::is_const<Atomic>{}, "");
    assert(bit < (sizeof(Integer) * 8));

    // otherwise default to the default implementation using fetch_and()
    return atomic_fetch_reset_default(atomic, bit, mo);
}

inline std::memory_order default_failure_memory_order(
    std::memory_order successMode) {
    switch (successMode) {
    case std::memory_order_acq_rel:
        return std::memory_order_acquire;
    case std::memory_order_release:
        return std::memory_order_relaxed;
    default:
        return successMode;
    }
}

} // namespace detail
} // namespace folly
