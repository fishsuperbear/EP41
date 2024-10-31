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
 * @file NCAtomic.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCATOMIC_H_
#define INCLUDE_NCORE_NCATOMIC_H_

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <atomic>

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
/*
 * A handful of basic atomic operations.  The appropriate pthread
 * functions should be used instead of these whenever possible.
 *
 * The "acquire" and "release" terms can be defined intuitively in terms
 * of the placement of memory barriers in a simple lock implementation:
 *   - wait until compare-and-swap(lock-is-free --> lock-is-held) succeeds
 *   - barrier
 *   - [do work]
 *   - barrier
 *   - store(lock-is-free)
 * In very crude terms, the initial (acquire) barrier prevents any of the
 * "work" from happening before the lock is held, and the later (release)
 * barrier ensures that all of the work happens before the lock is released.
 * (Think of cached writes, cache read-ahead, and instruction reordering
 * around the CAS and store instructions.)
 *
 * The barriers must apply to both the compiler and the CPU.  Note it is
 * legal for instructions that occur before an "acquire" barrier to be
 * moved down below it, and for instructions that occur after a "release"
 * barrier to be moved up above it.
 *
 * The ARM-driven implementation we use here is short on subtlety,
 * and actually requests a full barrier from the compiler and the CPU.
 * The only difference between acquire and release is in whether they
 * are issued before or after the atomic operation with which they
 * are associated.  To ease the transition to C/C++ atomic intrinsics,
 * you should not rely on this, and instead assume that only the minimal
 * acquire/release protection is provided.
 *
 * NOTE: all int32_t* values are expected to be aligned on 32-bit boundaries.
 * If they are not, atomicity is not guaranteed.
 */

/*
 * Basic arithmetic and bitwise operations.  These all provide a
 * barrier with "release" ordering, and return the previous value.
 *
 * These have the same characteristics (e.g. what happens on overflow)
 * as the equivalent non-atomic C operations.
 */
typedef std::atomic<int_least32_t> atomic_int_least32_t;

/**
 * @brief Addition of atomic operations,it will add 1 to addr
 *
 * @param addr addend
 * @return INT32 the result of addition
 */
INT32 nc_atomic_inc( volatile INT32 *const addr );

/**
 * @brief Subtraction of atomic operations,it will subtract 1.
 *
 * @param addr minute
 * @return INT32 the result of subtraction
 */
INT32 nc_atomic_dec( volatile INT32 *const addr );

/**
 * @brief Addition of atomic operations,it will add value to addr
 *
 * @param value addend
 * @param addr Summand
 * @return INT32 the result of addition
 */
INT32 nc_atomic_add( INT32 value, volatile INT32 *const addr );

/**
 * @brief Logical and of atomic operations,it will and value to addr
 *
 * @param value Operand
 * @param addr  Operand
 * @return INT32 the result of Logical and
 */
INT32 nc_atomic_and( INT32 value, volatile INT32 *const addr );

/**
 * @brief Logical or of atomic operations,it will and value to addr
 *
 * @param value Operand
 * @param addr Operand
 * @return INT32 the result of Logical and
 */
INT32 nc_atomic_or( INT32 value, volatile INT32 *const addr );

/*
 * Perform an atomic load with "acquire" or "release" ordering.
 *
 * This is only necessary if you need the memory barrier.  A 32-bit read
 * from a 32-bit aligned address is atomic on all supported platforms.
 */

/**
 * @brief load of atomic operations,it will return the value form addr pointer
 *
 * @param addr pointer of data
 * @return INT32 the result of load
 */
INT32 nc_atomic_acquire_load( const volatile INT32 *const addr );

/**
 * @brief load of atomic operations,it will return the value form addr pointer
 *        but no guarantee of execution order
 * @param addr pointer of data
 * @return INT32 the result of load
 */
INT32 nc_atomic_release_load( const volatile INT32 *addr );

/*
 * Perform an atomic store with "acquire" or "release" ordering.
 *
 * This is only necessary if you need the memory barrier.  A 32-bit write
 * to a 32-bit aligned address is atomic on all supported platforms.
 */

/**
 * @brief Perform an atomic store with "acquire" ordering.
 *
 * @param value data
 * @param addr pointer of address
 * @return VOID
 */
VOID nc_atomic_acquire_store( INT32 value, volatile INT32 *const addr );

/**
 * @brief Perform an atomic store with "release" ordering.
 *
 * @param value data
 * @param addr pointer of address
 * @return VOID
 */
VOID nc_atomic_release_store( INT32 value, volatile INT32 *const addr );
/*
 * Compare-and-set operation with "acquire" or "release" ordering.
 *
 * This returns one if the new value was successfully stored, which will
 * only happen when *addr == oldvalue.
 *
 * Implementations that use the release CAS in a loop may be less efficient
 * than possible, because we re-issue the memory barrier on each iteration.
 */

/**
 * @brief Compare-and-set operation with "acquire" ordering.
 *
 * @param oldvalue pointer to an value
 * @param newvalue value to copy to the contained value
 * @param addr pointer to an atomic value
 * @return INT32 0:the value contained by addr is equal the oldvalue
 *               1:others
 */
INT32 nc_atomic_acquire_cas( INT32 oldvalue, INT32 newvalue, volatile INT32 *const addr );

/**
 * @brief Compare-and-set operation with "release" ordering.
 *
 * @param oldvalue pointer to an value
 * @param newvalue value to copy to the contained value
 * @param addr pointer to an atomic value
 * @return INT32 0:the value contained by addr is equal the oldvalue
 *               1:others
 */
INT32 nc_atomic_release_cas( INT32 oldvalue, INT32 newvalue, volatile INT32 *const addr );
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCATOMIC_H_
/* EOF */
