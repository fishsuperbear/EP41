#pragma once

#include <atomic>

namespace hozon {
namespace netaos {
namespace adf {
template <typename T>
class Optional {
   public:
    Optional() : _has_value(false) {}

    Optional(const Optional& another) : _value(another._value) {
        _has_value.store(another._has_value.load(std::memory_order::memory_order_acquire),
                         std::memory_order::memory_order_release);
    }

    Optional(Optional&& another) : _value(another._value) {
        _has_value.store(another._has_value.load(std::memory_order::memory_order_acquire),
                         std::memory_order::memory_order_release);
    }

    Optional& operator=(const Optional& another) {
        _value = another._value;
        _has_value.store(another._has_value.load(std::memory_order::memory_order_acquire),
                         std::memory_order::memory_order_release);

        return *this;
    }

    Optional& operator=(Optional&& another) {
        _value = another._value;
        _has_value.store(another._has_value.load(std::memory_order::memory_order_acquire),
                         std::memory_order::memory_order_release);

        return *this;
    }

    Optional& operator=(T&& value) {
        _value = value;
        _has_value.store(true, std::memory_order::memory_order_release);

        return *this;
    }

    bool HasValue() { return _has_value.load(std::memory_order::memory_order_acquire); }

    T& Value() {
        _has_value.store(true, std::memory_order::memory_order_release);
        return _value;
    }

   private:
    T _value;
    std::atomic<bool> _has_value;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon