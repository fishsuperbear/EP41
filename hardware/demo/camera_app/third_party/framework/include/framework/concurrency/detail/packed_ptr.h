#pragma once

#include <stdint.h>
#include <assert.h>
#include <type_traits>
#include <ostream>

namespace netaos {
namespace framework {
namespace concurrency {
namespace detail {

template <typename T>
class PackedPtr {
    using reference = typename std::add_lvalue_reference<T>::type;
    static constexpr int BITS_PTR = 48;
    static constexpr int BITS_EXTRA = sizeof(void*) * 8 - BITS_PTR;
    static constexpr uintptr_t MASK_PTR = -1ull >> BITS_EXTRA;
    
public:
    void init(T* init_ptr = nullptr, uint16_t init_extra = 0) {
        auto ptr = reinterpret_cast<uintptr_t>(init_ptr);
        assert(!(ptr >> BITS_PTR) && "the upper two bytes should be zero");
        _data = (uintptr_t(init_extra) << BITS_PTR) | ptr;
    }

    void set(T* t) {
        uintptr_t ptr = reinterpret_cast<uintptr_t>(t);
        assert(!(ptr >> BITS_PTR) && "the upper two bytes should be zero");
        _data >>= BITS_PTR;
        _data <<= BITS_PTR;
        _data |= ptr;
    }

    T* get() const {
        return reinterpret_cast<T*>(_data & MASK_PTR);
    }

    T* operator->() const {
        return get();
    }

    reference operator*() const {
        return *get();
    }

    reference operator[](std::ptrdiff_t i) const {
        return get()[i];
    }

    uint16_t extra() const {
        return _data >> BITS_PTR;
    }

    void set_extra(uint16_t extra) {
        _data &= MASK_PTR;
        _data |= (uintptr_t(extra) << BITS_PTR);
    }

private:
    uintptr_t _data;
} __attribute__((__packed__));

static_assert(
    std::is_pod<PackedPtr<void>>::value, "PackedPtr must be kept a POC type"
);

static_assert(
    sizeof(PackedPtr<void>) == 8, "PackedPtr should be only 8 bytes"
);

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const PackedPtr<T>& p) {
    os << "PackedPtr(" << p.get() << ", " << p.extra() << ")";
    return os;
}

} // namespace detail
} // namespace concurrency
} // namespace framework
} // namespace netaos