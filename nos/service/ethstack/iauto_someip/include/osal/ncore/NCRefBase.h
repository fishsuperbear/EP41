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
 * @file NCRefBase.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCREFBASE_H_
#define INCLUDE_NCORE_NCREFBASE_H_

#include <memory>

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// #define COMPARE(_op_)
// inline bool operator _op_(const sp<T>& o) const {
// return m_sptr.get() _op_ o.m_sptr.get();
// }
// inline bool operator _op_(const T* o) const {
// return m_sptr.get() _op_ o;
// }
// template<typename U>
// inline bool operator _op_(const sp<U>& o) const {
// return m_sptr.get() _op_ o.m_sptr.get();
// }
// template<typename U>
// inline bool operator _op_(const U* o) const {
// return m_sptr.get() _op_ o;
// }
// inline bool operator _op_(const wp<T>& o) const {
// return m_sptr.get() _op_ o.m_wptr.lock().get();
// }
// template<typename U>
// inline bool operator _op_(const wp<U>& o) const {
// return m_sptr.get() _op_ o.m_wptr.lock().get();
// }

// #define COMPARE_WEAK(_op_)
// inline bool operator _op_(const sp<T>& o) const {
// return m_wptr.lock().get() _op_ o.m_sptr.get();
// }
// inline bool operator _op_(const T* o) const {
// return m_wptr.lock().get() _op_ o;
// }
// template<typename U>
// inline bool operator _op_(const sp<U>& o) const {
// return m_wptr.lock().get() _op_ o.m_sptr.get();
// }
// template<typename U>
// inline bool operator _op_(const U* o) const {
// return m_wptr.lock().get() _op_ o;
// }

template <typename T>
class wp;

/**
 * sample code:
 * @code
 * class A
 * {
 * };
 *
 * //load a object with a strong pointer.
 * sp<A> spa = new A();
 *
 * //reload with a weak pointer.
 * wp<A> wpa = spa;
 *
 * //promote the weak pointer to a strong pointer.
 * //it fails if the obj already destoryed.
 * sp<A> spa2 = wpa.promote();
 *
 * @endcode
 */
template <typename T>
class sp {
   public:
    /**
     * @brief Construct a new sp
     *
     */
    inline sp() : m_sptr( nullptr ) {}

    /**
     * @brief Construct a new sp
     *
     * @param other T type poniter
     */
    sp( T *const other ) : m_sptr( other ) {}

    /**
     * @brief Construct a new sp
     *
     * @param other  T type reference
     */
    sp( const sp<T> &other ) : m_sptr( other.m_sptr ) {}

    /**
     * @brief Construct a new sp
     *
     * @param other T type reference
     */
    sp( sp<T> &&other ) : m_sptr( other.m_sptr ) {}

    /**
     * @brief Construct a new sp
     *
     * @param other U type pointer
     */
    template <typename U>
    sp( U *const other ) : m_sptr( static_cast<T *>( other ) ) {}

    /**
     * @brief Construct a new sp
     *
     * @param other T type reference
     */
    template <typename U>
    sp( const sp<U> &other ) : m_sptr( std::static_pointer_cast<T>( other.m_sptr ) ) {}

    /**
     * @brief Construct a new sp
     *
     * @param other T type reference
     */
    template <typename U>
    sp( sp<U> &&other ) : m_sptr( std::static_pointer_cast<T>( other.m_sptr ) ) {}

    /**
     * @brief Destroy the sp
     */
    ~sp() {}

    /**
     * @brief get tht T type reference
     *
     * @return T& T type reference
     */
    inline T &operator*() const { return *( m_sptr.get() ); }

    /**
     * @brief get the T type pointer
     *
     * @return T* T type pointer
     */
    inline T *operator->() const { return m_sptr.get(); }

    /**
     * @brief get the T type pointer
     *
     * @return T* T type pointer
     */
    inline T *get() const { return m_sptr.get(); }

    /**
     * @brief assignment function create a new sp from the "other" T type pointer
     *
     * @param other T type pointer
     * @return sp&  new sp
     */
    sp &operator=( T *const other ) {
        if ( m_sptr.get() != other ) {
            m_sptr = std::shared_ptr<T>( other );
        }
        return *this;
    }

    /**
     * @brief assignment function create a new sp from the "other" sp
     *
     * @param other sp
     * @return sp& new sp
     */
    sp &operator=( const sp<T> &other ) {
        if ( m_sptr.get() != other.get() ) {
            m_sptr = other.m_sptr;
        }
        return *this;
    }

    /**
     * @brief assignment function create a new sp from the "other"
     *
     * @param other sp
     * @return sp& new sp
     */
    sp &operator=( sp<T> &&other ) {
        if ( m_sptr.get() != other.get() ) {
            m_sptr = other.m_sptr;
        }
        return *this;
    }

    /**
     * @brief  assignment function create a new sp from the "other"
     *
     * @param other sp
     * @return sp& new sp
     */
    template <typename U>
    sp &operator=( U *const other ) {
        if ( m_sptr.get() != other ) {
            m_sptr = std::shared_ptr<T>( static_cast<T *>( other ) );
        }
        return *this;
    }

    /**
     * @brief assignment function create a new sp from the "other"
     *
     * @param other sp
     * @return sp&  new sp
     */
    template <typename U>
    sp &operator=( const sp<U> &other ) {
        if ( m_sptr.get() != other.get() ) {
            m_sptr = std::static_pointer_cast<T>( other.m_sptr );
        }
        return *this;
    }

    /**
     * @brief assignment function create a new sp from the "other"
     *
     * @param other sp
     * @return sp&  new sp
     */
    template <typename U>
    sp &operator=( sp<U> &&other ) {
        if ( m_sptr.get() != other.get() ) {
            m_sptr = std::static_pointer_cast<T>( other.m_sptr );
        }
        return *this;
    }

    /**
     * @brief reset
     */
    void clear() { m_sptr.reset(); }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o sp object
     * @return true the condition holds
     * @return false otherwise
     */
    inline bool operator==( const sp<T> &o ) const { return m_sptr.get() == o.m_sptr.get(); }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o T type pointer
     * @return true the condition holds
     * @return false otherwise
     */
    inline bool operator==( const T *const o ) const { return m_sptr.get() == o; }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o sp object
     * @return true the condition holds
     * @return false otherwise
     */
    template <typename U>
    inline bool operator==( const sp<U> &o ) const {
        return m_sptr.get() == o.m_sptr.get();
    }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o U type pointer
     * @return true the condition holds
     * @return false otherwise
     */
    template <typename U>
    inline bool operator==( const U *const o ) const {
        return m_sptr.get() == o;
    }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o wp object
     * @return true the condition holds
     * @return false otherwise
     */
    inline bool operator==( const wp<T> &o ) const { return m_sptr.get() == o.m_wptr.lock().get(); }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o wp object
     * @return true the condition holds
     * @return false otherwise
     */
    template <typename U>
    inline bool operator==( const wp<U> &o ) const {
        return m_sptr.get() == o.m_wptr.lock().get();
    }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o sp object
     * @return true the condition holds
     * @return false otherwise
     */
    inline bool operator!=( const sp<T> &o ) const { return m_sptr.get() != o.m_sptr.get(); }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o T type pointer
     * @return true the condition holds
     * @return false otherwise
     */
    inline bool operator!=( const T *const o ) const { return m_sptr.get() != o; }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o sp object
     * @return true the condition holds
     * @return false otherwise
     */
    template <typename U>
    inline bool operator!=( const sp<U> &o ) const {
        return m_sptr.get() != o.m_sptr.get();
    }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o U type pointer
     * @return true the condition holds
     * @return false otherwise
     */
    template <typename U>
    inline bool operator!=( const U *const o ) const {
        return m_sptr.get() != o;
    }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o wp object
     * @return true the condition holds
     * @return false otherwise
     */
    inline bool operator!=( const wp<T> &o ) const { return m_sptr.get() != o.m_wptr.lock().get(); }

    /**
     * @brief Performs the appropriate relational comparison operation between the shared_ptr
     * objects lhs and rhs.
     *
     * @param o wp object
     * @return true the condition holds
     * @return false otherwise
     */
    template <typename U>
    inline bool operator!=( const wp<U> &o ) const {
        return m_sptr.get() != o.m_wptr.lock().get();
    }

   private:
    template <typename Y>
    friend class sp;
    template <typename Y>
    friend class wp;
    template <typename Y>
    friend class enable_shared_from_this;
    std::shared_ptr<T> m_sptr;
};

/**
 * @brief weak point
 *
 * @class wp
 */
template <typename T>
class wp {
   public:
    inline wp() : m_wptr() {}

    wp( const wp<T> &other ) : m_wptr( other.m_wptr ) {}

    wp( const sp<T> &other ) : m_wptr( other.m_sptr ) {}

    template <typename U>
    wp( const sp<U> &other ) : m_wptr( std::static_pointer_cast<T>( other.m_sptr ) ) {}

    template <typename U>
    wp( const wp<U> &other ) : m_wptr() {
        const std::shared_ptr<U> otherPtr = other.m_wptr.lock();
        m_wptr                            = std::static_pointer_cast<T>( otherPtr );
    }

    ~wp() {}

    // Assignment
    wp &operator=( const wp<T> &other ) {
        m_wptr = other.m_wptr;
        return *this;
    }

    wp &operator=( const sp<T> &other ) {
        m_wptr = other.m_sptr;
        return *this;
    }

    template <typename U>
    wp &operator=( const sp<U> &other ) {
        m_wptr = std::static_pointer_cast<T>( other.m_sptr );
        return *this;
    }

    template <typename U>
    wp &operator=( const wp<U> &other ) {
        const std::shared_ptr<U> otherPtr = other.m_wptr.lock();
        m_wptr                            = std::static_pointer_cast<T>( otherPtr );
        return *this;
    }

    // promotion to sp
    sp<T> promote() const {
        sp<T> t_result;
        t_result.m_sptr = this->m_wptr.lock();
        return t_result;
    }

    // Reset
    void clear() { m_wptr.reset(); }

    // expired
    bool expired() const { return m_wptr.expired(); }

    inline bool operator==( const sp<T> &o ) const { return m_wptr.lock().get() == o.m_sptr.get(); }

    inline bool operator==( const T *const o ) const { return m_wptr.lock().get() == o; }

    template <typename U>
    inline bool operator==( const sp<U> &o ) const {
        return m_wptr.lock().get() == o.m_sptr.get();
    }

    template <typename U>
    inline bool operator==( const U *const o ) const {
        return m_wptr.lock().get() == o;
    }

    inline bool operator!=( const sp<T> &o ) const { return m_wptr.lock().get() != o.m_sptr.get(); }

    inline bool operator!=( const T *const o ) const { return m_wptr.lock().get() != o; }

    template <typename U>
    inline bool operator!=( const sp<U> &o ) const {
        return m_wptr.lock().get() != o.m_sptr.get();
    }

    template <typename U>
    inline bool operator!=( const U *const o ) const {
        return m_wptr.lock().get() != o;
    }

    inline bool operator==( const wp<T> &o ) const {
        return ( m_wptr.lock().get() == o.m_wptr.lock().get() );
    }

    template <typename U>
    inline bool operator==( const wp<U> &o ) const {
        return m_wptr.lock().get() == o.m_wptr.lock().get();
    }

    inline bool operator!=( const wp<T> &o ) const {
        return m_wptr.lock().get() != o.m_wptr.lock().get();
    }

    template <typename U>
    inline bool operator!=( const wp<U> &o ) const {
        return !operator==( o );
    }

   private:
    template <typename Y>
    friend class sp;
    template <typename Y>
    friend class wp;
    std::weak_ptr<T> m_wptr;
};

template <typename T>
using sptr = sp<T>;

template <typename T>
using wptr = wp<T>;

template <typename T>
struct ncsp {
    typedef sptr<T> sp;
    typedef wptr<T> wp;
};

template <typename T>
class enable_shared_from_this : public std::enable_shared_from_this<T> {
   public:
    sp<T> sp_from_this() {
        sp<T> t_result;
        t_result.m_sptr = std::enable_shared_from_this<T>::shared_from_this();
        return t_result;
    }
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCREFBASE_H_
/* EOF */
