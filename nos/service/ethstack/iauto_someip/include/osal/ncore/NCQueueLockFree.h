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
 * @file NCQueueLockFree.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCQUEUELOCKFREE_H_
#define INCLUDE_NCORE_NCQUEUELOCKFREE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "osal/ncore/NCAtomicPtr.h"
#include "osal/ncore/NCQueue.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
/**
 * @template class NCQueueLockFree
 *
 * @brief  Lock-free queue implementation.
 *
 * Only a single thread can read or write from the pipe
 *      at any specific moment.
 *
 * T is the type of the object in the queue.
 * N is granularity of the pipe, i.e. how many items are needed to
 *   perform next memory allocation.
 */
template <typename T, const INT32 N>
class NCQueueLockFree {
   public:
    /**
     * @brief Construct a new NCQueueLockFree
     */
    inline NCQueueLockFree() {
        //  Insert terminator element into the queue.
        m_queue.push();

        //  Let all the pointers to point to the terminator.
        //  (unless pipe is dead, in which case c is set to NULL).
        m_f = &m_queue.back();
        m_w = m_f;
        m_r = m_w;
        m_c.set( &m_queue.back() );
    }

    /**
     * @brief Destroy the NCQueueLockFree
     */
    inline virtual ~NCQueueLockFree() {}

    //  Write an item to the pipe.  Don't flush it yet. If incomplete is
    //  set to NC_TRUE the item is assumed to be continued by items
    //  subsequently written to the pipe. Incomplete items are never
    //  flushed down the stream.

    /**
     * @brief Write an item to the pipe.
     *        Don't flush it yet.
     *        If incomplete is set to NC_TRUE the item is assumed to be continued by items
     *        subsequently written to the pipe.
     *        Incomplete items are never flushed down the stream.
     *
     * @param value_ the data want to write
     * @param incomplete_ NC_TRUE the item is assumed to be continued by items
     *        subsequently written to the pipe.
     */
    inline void writeQueue( const T &value_, NC_BOOL incomplete_ ) {
        //  Place the value to the queue, add new terminator element.
        m_queue.back() = value_;
        m_queue.push();

        //  Move the "flush up to here" poiter.
        if ( !incomplete_ ) {
            m_f = &m_queue.back();
        }
    }

    /**
     * @brief Pop an incomplete item from the pipe.
     *
     * @param value_ get the data from the pipe
     * @return NC_BOOL Returns NC_TRUE is such item exists, NC_FALSE otherwise.
     */
    inline NC_BOOL unwrite( T *value_ ) {
        if ( m_f == &m_queue.back() ) {
            return NC_FALSE;
        }
        m_queue.unpush();
        *value_ = m_queue.back();
        return NC_TRUE;
    }

    /**
     * @brief Flush all the completed items into the pipe.
     *
     * @return NC_BOOL NC_FALSE if the reader thread is sleeping.
     *                          In that case, caller is obliged to
     *                          wake the reader up before using the pipe again.
     */
    inline NC_BOOL flush() {
        //  If there are no un-flushed items, do nothing.
        if ( m_w == m_f ) {
            return NC_TRUE;
        }

        //  Try to set 'c' to 'f'.
        if ( m_c.cas( m_w, m_f ) != m_w ) {
            //  Compare-and-swap was unseccessful because 'c' is NULL.
            //  This means that the reader is asleep. Therefore we don't
            //  care about thread-safeness and update c in non-atomic
            //  manner. We'll return NC_FALSE to let the caller know
            //  that reader is sleeping.
            m_c.set( m_f );
            m_w = m_f;
            return NC_FALSE;
        }

        //  Reader is alive. Nothing special to do now. Just move
        //  the 'first un-flushed item' pointer to 'f'.
        m_w = m_f;
        return NC_TRUE;
    }

    /**
     * @brief Check whether item is available for reading.
     *
     * @return NC_BOOL NC_TRUE:read available NC_FALSE:otherwise
     */
    inline NC_BOOL checkRead() {
        //  Was the value prefetched already? If so, return.
        if ( ( &m_queue.front() != m_r ) && ( m_r != nullptr ) ) {
            return NC_TRUE;
        }

        //  There's no prefetched value, so let us prefetch more values.
        //  Prefetching is to simply retrieve the
        //  pointer from c in atomic fashion. If there are no
        //  items to prefetch, set c to NULL (using compare-and-swap).
        m_r = m_c.cas( &m_queue.front(), nullptr );

        //  If there are no elements prefetched, exit.
        //  During pipe's lifetime r should never be NULL, however,
        //  it can happen during pipe shutdown when items
        //  are being deallocated.
        if ( ( &m_queue.front() == m_r ) || ( m_r == nullptr ) ) {
            return NC_FALSE;
        }

        //  There was at least one value prefetched.
        return NC_TRUE;
    }

    /**
     * @brief Reads an item from the pipe.
     *
     * @param value_ data to get value
     * @return NC_BOOL NC_FALSE if there is no value
     */
    inline NC_BOOL readQueue( T *value_ ) {
        //  Try to prefetch a value.
        if ( !checkRead() ) {
            return NC_FALSE;
        }

        //  There was at least one value prefetched.
        //  Return it to the caller.
        *value_ = m_queue.front();
        m_queue.pop();
        return NC_TRUE;
    }

    //  Applies the function fn to the first elemenent in the pipe
    //  and returns the value returned by the fn.
    //  The pipe mustn't be empty or the function crashes.

    /**
     * @brief Applies the function fn to the first elemenent in the pipe
     *        and returns the value returned by the fn.
     *        The pipe mustn't be empty or the function crashes.
     *
     * @param fn function pointer
     * @return NC_BOOL the operation result of fn
     */
    inline NC_BOOL probe( NC_BOOL ( *fn )( T & ) ) {
        const NC_BOOL rc = checkRead();
        assert( rc );

        return ( *fn )( m_queue.front() );
    }

   protected:
    //  Allocation-efficient queue to store pipe items.
    //  Front of the queue points to the first prefetched item, back of
    //  the pipe points to last un-flushed item. Front is used only by
    //  reader thread, while back is used only by writer thread.
    NCQueue<T, N> m_queue;

    //  Points to the first un-flushed item. This variable is used
    //  exclusively by writer thread.
    T *m_w;

    //  Points to the first un-prefetched item. This variable is used
    //  exclusively by reader thread.
    T *m_r;

    //  Points to the first item to be flushed in the future.
    T *m_f;

    //  The single point of contention between writer and reader thread.
    //  Points past the last flushed item. If it is NULL,
    //  reader is asleep. This pointer should be always accessed using
    //  atomic operations.
    NCAtomicPtr<T> m_c;

   protected:
    // Disable copy construction and assignment.
    NCQueueLockFree( const NCQueueLockFree & );
    const NCQueueLockFree &operator=( const NCQueueLockFree & );
};
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCQUEUELOCKFREE_H_
/* EOF */
