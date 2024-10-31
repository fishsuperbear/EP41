#pragma once

#include <limits.h>
#include <assert.h>
#include <memory>
#include <atomic>

namespace netaos {
namespace framework {
namespace concurrency {
namespace detail {

class CountedDetail final {
public:
    using shared_count = std::__shared_count<std::_S_atomic>;
    using counted_base = std::_Sp_counted_base<std::_S_atomic>;

private:
    struct AccessSharedPtr {
        typedef shared_count (std::__shared_ptr<const void, std::_S_atomic>::*type);
        friend type field_ptr(AccessSharedPtr);
    };

    struct AccessBase {
        typedef counted_base* (shared_count::*type);
        friend type field_ptr(AccessBase);
    };

    struct AccessUseCount {
        typedef _Atomic_word (counted_base::*type);
        friend type field_ptr(AccessUseCount);
    };

    struct AccessWeakCount {
        typedef _Atomic_word (counted_base::*type);
        friend type field_ptr(AccessWeakCount);
    };

    struct AccessCountedPtrPtr {
        typedef const void* (std::_Sp_counted_ptr<const void*, std::_S_atomic>::*type);
        friend type field_ptr(AccessCountedPtrPtr);
    };

    struct AccessSharedPtrPtr {
        typedef const void* (std::__shared_ptr<const void, std::_S_atomic>::*type);
        friend type field_ptr(AccessSharedPtrPtr);
    };

    struct AccessRefCount {
        typedef shared_count (std::__shared_ptr<const void, std::_S_atomic>::*type);
        friend type field_ptr(AccessRefCount);
    };

    template <typename Tag, typename Tag::type M>
    struct Rob {
        friend typename Tag::type field_ptr(Tag) {
            return M;
        }
    };

public:

    template <typename T, typename... Args>
    static std::shared_ptr<T> make_ptr(Args&&... args) {
        return std::make_shared<T>(std::forward<Args...>(args...));
    }

    template <typename T>
    static counted_base* get_counted_base(const std::shared_ptr<T>& bar) {
        const std::shared_ptr<const void>& ptr(
            reinterpret_cast<const std::shared_ptr<const void>&>(bar));
        return (ptr.*field_ptr(AccessSharedPtr{})).*field_ptr(AccessBase{});
    }

    template <typename T>
    static T* get_shared_ptr(counted_base* base) {
        auto inplace = base->_M_get_deleter(typeid(std::_Sp_make_shared_tag));
        if (inplace) {
            return (T*)inplace;
        }
        using derived_type = std::_Sp_counted_ptr<const void*, std::_S_atomic>;
        auto ptr = reinterpret_cast<derived_type*>(base);
        return (T*)(ptr->*field_ptr(AccessCountedPtrPtr{}));
    }

    static void inc_shared_count(counted_base* base, long count) {
        assert((base->_M_get_use_count() + count < INT_MAX) && "atomic_shared_ptr overflow");
        __gnu_cxx::__atomic_add_dispatch(&(base->*field_ptr(AccessUseCount{})), count);
    }

    template <typename T>
    static void release_shared(counted_base* base, long count) {
        if (__gnu_cxx::__exchange_and_add_dispatch(
            &(base->*field_ptr(AccessUseCount{})), -count) == count) {
            base->_M_dispose();

            if (__gnu_cxx::__exchange_and_add_dispatch(
                    &(base->*field_ptr(AccessWeakCount{})), -1) == 1) {
                base->_M_destroy();
            }
        }
    }

    template <typename T>
    static T* release_ptr(std::shared_ptr<T>& p) {
        auto res = p.get();
        std::shared_ptr<const void>& ptr(reinterpret_cast<std::shared_ptr<const void>&>(p));
        ptr.*field_ptr(AccessSharedPtrPtr{}) = nullptr;
        (ptr.*field_ptr(AccessRefCount{})).*field_ptr(AccessBase{}) = nullptr;
        return res;
    }

    template <typename T>
    static std::shared_ptr<T> get_shared_ptr_from_counted_base(counted_base* base, bool inc = true) {
        if (!base) {
            return nullptr;
        }
        std::shared_ptr<const void> newp;
        if (inc) {
            inc_shared_count(base, 1);
        }
        newp.*field_ptr(AccessSharedPtrPtr{}) = get_shared_ptr<const void>(base);
        (newp.*field_ptr(AccessRefCount{})).*field_ptr(AccessBase{}) = base;
        auto res = reinterpret_cast<std::shared_ptr<T>*>(&newp);
        return std::move(*res);
    }

    static long get_use_count(counted_base* base) {
        return base->_M_get_use_count();
    }

};

template struct CountedDetail::Rob<
    CountedDetail::AccessSharedPtr, &std::__shared_ptr<const void, std::_S_atomic>::_M_refcount>;

template struct CountedDetail::Rob<
    CountedDetail::AccessBase, &CountedDetail::shared_count::_M_pi>;

template struct CountedDetail::Rob<
    CountedDetail::AccessUseCount, &CountedDetail::counted_base::_M_use_count>;

template struct CountedDetail::Rob<
    CountedDetail::AccessWeakCount, &CountedDetail::counted_base::_M_weak_count>;

template struct CountedDetail::Rob<
    CountedDetail::AccessCountedPtrPtr, &std::_Sp_counted_ptr<const void*, std::_S_atomic>::_M_ptr>;

template struct CountedDetail::Rob<
    CountedDetail::AccessSharedPtrPtr, &std::__shared_ptr<const void, std::_S_atomic>::_M_ptr>;

template struct CountedDetail::Rob<
    CountedDetail::AccessRefCount, &std::__shared_ptr<const void, std::_S_atomic>::_M_refcount>;

} // namespace detail
} // namespace concurrency
} // namespace framework
} // namespace netaos
