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
 * @file NCAtomicPtr.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCATOMICPTR_H_
#define INCLUDE_NCORE_NCATOMICPTR_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#if ( defined __i386__ || defined __x86_64__ ) && defined __GNUC__
#define NC_ATOMIC_PTR_X86
#elif defined __ARM_ARCH_7A__ && defined __GNUC__
#define NC_ATOMIC_PTR_ARM
#elif ( defined NC_HAVE_SOLARIS || defined NC_HAVE_NETBSD )
#define NC_ATOMIC_PTR_ATOMIC_H
#else
#define NC_ATOMIC_PTR_MUTEX
#endif
#include <stddef.h>

#include "osal/ncore/NCNameSpace.h"
#if defined NC_ATOMIC_PTR_ATOMIC_H
#include <atomic.h>
#elif defined NC_ATOMIC_PTR_MUTEX
#include "osal/ncore/NCSyncObj.h"
#endif

OSAL_BEGIN_NAMESPACE
/**
 * @template class NCAtomicPtr
 *
 * @brief This class encapsulates several atomic operations on pointers.
 */
template <typename T>
class NCAtomicPtr {
   public:
    /**
     * @brief Construct a new NCAtomicPtr object
     *
     */
    inline NCAtomicPtr() { m_ptr = nullptr; }

    /**
     * @brief Destroy the NCAtomicPtr object
     *
     */
    inline ~NCAtomicPtr() {}

    //  Set value of atomic pointer in a non-threadsafe way
    //  Use this function only when you are sure that at most one
    //  thread is accessing the pointer at the moment.

    /**
     * @brief Set value of atomic pointer in a non-threadsafe way
     *        Use this function only when you are sure that at most one
     *        thread is accessing the pointer at the moment.
     *
     * @param ptr_ pointer of object
     */
    inline void set( T *ptr_ ) { this->m_ptr = ptr_; }

    /**
     * @brief Perform atomic 'exchange pointers' operation. Pointer is set
     *        to the 'val' value. Old value is returned.
     * @param val_ new pointer of object T
     * @return T* Old value of m_ptr,before changing
     */
    inline T *xchg( T *val_ ) {
#if defined NC_ATOMIC_PTR_ATOMIC_H
        return reinterpret_cast<T *>( atomic_swap_ptr( &m_ptr, val_ ) );
#elif defined NC_ATOMIC_PTR_X86
        T *old;
        __asm__ volatile( "lock; xchg %0, %2"
                          : "=r"( old ), "=m"( m_ptr )
                          : "m"( m_ptr ), "0"( val_ ) );
        return old;
#elif defined NC_ATOMIC_PTR_ARM
        T *          old;
        unsigned int flag;
        __asm__ volatile(
            "       dmb     sy\n\t"
            "1:     ldrex   %1, [%3]\n\t"
            "       strex   %0, %4, [%3]\n\t"
            "       teq     %0, #0\n\t"
            "       bne     1b\n\t"
            "       dmb     sy\n\t"
            : "=&r"( flag ), "=&r"( old ), "+Qo"( m_ptr )
            : "r"( &m_ptr ), "r"( val_ )
            : "cc" );
        return old;
#elif defined NC_ATOMIC_PTR_MUTEX
        m_sync.syncStart();
        T *const old = const_cast<T *const>( m_ptr );
        m_ptr        = val_;
        m_sync.syncEnd();
        return old;
#else
#error atomic_ptr is not implemented for this platform
#endif
    }

    //  Perform atomic 'compare and swap' operation on the pointer.
    //  The pointer is compared to 'cmp' argument and if they are
    //  equal, its value is set to 'val'. Old value of the pointer
    //  is returned.

    /**
     * @brief Perform atomic 'compare and swap' operation on the pointer.
     *        The pointer is compared to 'cmp' argument and if they are
     *        equal, its value is set to 'val'. Old value of the pointer
     *        is returned.
     * @param cmp_ [IN] pointer of new value
     * @param val_ [OUT] value for getting
     * @return T* Old value of the pointer
     */
    inline T *cas( const T *const cmp_, T *val_ ) {
#if defined NC_ATOMIC_PTR_ATOMIC_H
        return reinterpret_cast<T *>( atomic_cas_ptr( &m_ptr, cmp_, val_ ) );
#elif defined NC_ATOMIC_PTR_X86
        T *old;
        __asm__ volatile( "lock; cmpxchg %2, %3"
                          : "=a"( old ), "=m"( m_ptr )
                          : "r"( val_ ), "m"( m_ptr ), "0"( cmp_ )
                          : "cc" );
        return old;
#elif defined NC_ATOMIC_PTR_ARM
        T *          old;
        unsigned int flag;
        __asm__ volatile(
            "       dmb     sy\n\t"
            "1:     ldrex   %1, [%3]\n\t"
            "       mov     %0, #0\n\t"
            "       teq     %1, %4\n\t"
            "       it      eq\n\t"
            "       strexeq %0, %5, [%3]\n\t"
            "       teq     %0, #0\n\t"
            "       bne     1b\n\t"
            "       dmb     sy\n\t"
            : "=&r"( flag ), "=&r"( old ), "+Qo"( m_ptr )
            : "r"( &m_ptr ), "r"( cmp_ ), "r"( val_ )
            : "cc" );
        return old;
#elif defined NC_ATOMIC_PTR_MUTEX
        m_sync.syncStart();
        T *old = const_cast<T *>( m_ptr );
        if ( m_ptr == cmp_ ) {
            m_ptr = val_;
        }
        m_sync.syncEnd();
        return old;
#else
#error atomic_ptr is not implemented for this platform
#endif
    }

   private:
#if defined NC_ATOMIC_PTR_MUTEX
    NCSyncObj         m_sync;
    volatile T *m_ptr THREAD_ANNOTATION_ATTRIBUTE__( guarded_by( m_sync ) );
#else
    volatile T *m_ptr;
#endif

    /**
     * @brief Construct a new NCAtomicPtr object
     *        Disable copy construction and assignment
     *
     */
    NCAtomicPtr( const NCAtomicPtr & );

    /**
     * @brief Construct a new NCAtomicPtr object
     *        Disable copy construction and assignment
     *
     * @return const NCAtomicPtr&
     */
    const NCAtomicPtr &operator=( const NCAtomicPtr & );
};

OSAL_END_NAMESPACE

//  Remove macros local to this file.
#if defined NC_ATOMIC_PTR_ATOMIC_H
#undef NC_ATOMIC_PTR_ATOMIC_H
#endif
#if defined NC_ATOMIC_PTR_X86
#undef NC_ATOMIC_PTR_X86
#endif
#if defined NC_ATOMIC_PTR_ARM
#undef NC_ATOMIC_PTR_ARM
#endif
#if defined NC_ATOMIC_PTR_MUTEX
#undef NC_ATOMIC_PTR_MUTEX
#endif

#endif  // INCLUDE_NCORE_NCATOMICPTR_H_
/* EOF */
