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
 * @file NCQueue.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCQUEUE_H_
#define INCLUDE_NCORE_NCQUEUE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <stdint.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "osal/ncore/NCAtomicPtr.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
/**
 * @template class NCQueue
 *
 * @brief  an efficient queue implementation
 *
 * The main goal is to minimise number of allocations/deallocations needed.
 * Thus NCQueue allocates/deallocates elements in batches of N.
 *
 * T is the type of the object in the queue.
 * N is granularity of the queue (how many pushes have to be done till
 *   actual memory allocation is required).
 */
template <typename T, const INT32 N>
class NCQueue {
   public:
    /**
     * @brief Construct a new NCQueue object
     */
    inline NCQueue() {
        m_beginChunk = static_cast<chunk_t *>( malloc( sizeof( chunk_t ) ) );
        assert( m_beginChunk != static_cast<chunk_t *>( nullptr ) );
        m_beginPos  = 0;
        m_backChunk = nullptr;
        m_backPos   = 0;
        m_endChunk  = m_beginChunk;
        m_endPos    = 0;
    }

    /**
     * @brief Destroy the NCQueue object
     */
    inline ~NCQueue() {
        while ( true ) {
            if ( m_beginChunk == m_endChunk ) {
                free( m_beginChunk );
                m_beginChunk = nullptr;
                break;
            }
            chunk_t *o   = m_beginChunk;
            m_beginChunk = m_beginChunk->next;
            free( o );
            o = nullptr;
        }

        chunk_t *sc = reinterpret_cast<chunk_t *>( m_spareChunk.xchg( nullptr ) );
        if ( sc != nullptr ) {
            free( sc );
            sc = nullptr;
        }
    }

    /**
     * @brief  get the front element of the queue
     *
     * @return T& Returns reference to the front element of the queue.
     *         If the queue is empty, behaviour is undefined.
     */
    inline T &front() { return m_beginChunk->values[ m_beginPos ]; }

    /**
     * @brief get the back element of the queue.
     *
     * @return T& Returns reference to the back element of the queue.
     *         If the queue is empty, behaviour is undefined.
     */
    inline T &back() { return m_backChunk->values[ m_backPos ]; }

    /**
     * @brief Adds an element to the back end of the queue.
     */
    inline void push() {
        m_backChunk = m_endChunk;
        m_backPos   = m_endPos;
        ++m_endPos;
        if ( m_endPos != N ) {
            return;
        }

        chunk_t *sc = reinterpret_cast<chunk_t *>( m_spareChunk.xchg( nullptr ) );
        if ( sc != nullptr ) {
            m_endChunk->next = sc;
            sc->prev         = m_endChunk;
        } else {
            m_endChunk->next = static_cast<chunk_t *>( malloc( sizeof( chunk_t ) ) );
            assert( m_endChunk->next != static_cast<chunk_t *>( nullptr ) );
            m_endChunk->next->prev = m_endChunk;
        }
        m_endChunk = m_endChunk->next;
        m_endPos   = 0;
    }

    //  Removes element from the back end of the queue. In other words
    //  it rollbacks last push to the queue. Take care: Caller is
    //  responsible for destroying the object being unpushed.
    //  The caller must also guarantee that the queue isn't empty when
    //  unpush is called. It cannot be done automatically as the read
    //  side of the queue can be managed by different, completely
    //  unsynchronised thread.

    /**
     * @brief Removes element from the back end of the queue.
     *        In other words it rollbacks last push to the queue.
     *        Take care: Caller is responsible for destroying the object being unpushed.
     *        The caller must also guarantee that the queue isn't empty when
     *        unpush is called. It cannot be done automatically as the read
     *        side of the queue can be managed by different, completely
     *        unsynchronised thread.
     */
    inline void unpush() {
        //  First, move 'back' one position backwards.
        if ( m_backPos != 0 ) {
            --m_backPos;
        } else {
            m_backPos   = N - 1;
            m_backChunk = m_backChunk->prev;
        }

        //  Now, move 'end' position backwards. Note that obsolete end chunk
        //  is not used as a spare chunk. The analysis shows that doing so
        //  would require free and atomic operation per chunk deallocated
        //  instead of a simple free.
        if ( m_endPos != 0 ) {
            --m_endPos;
        } else {
            m_endPos   = N - 1;
            m_endChunk = m_endChunk->prev;
            free( m_endChunk->next );
            m_endChunk->next = nullptr;
        }
    }

    /**
     * @brief Removes an element from the front end of the queue.
     */
    inline void pop() {
        ++m_beginPos;
        if ( m_beginPos == N ) {
            chunk_t *const o   = m_beginChunk;
            m_beginChunk       = m_beginChunk->next;
            m_beginChunk->prev = nullptr;
            m_beginPos         = 0;

            //  'o' has been more recently used than spare_chunk,
            //  so for cache reasons we'll get rid of the spare and
            //  use 'o' as the spare.
            chunk_t *cs = m_spareChunk.xchg( o );
            if ( cs != nullptr ) {
                free( cs );
                cs = nullptr;
            }
        }
    }

   private:
    //  Individual memory chunk to hold N elements.
    struct chunk_t {
        T        values[ N ];
        chunk_t *prev;
        chunk_t *next;
    };

    //  Back position may point to invalid memory if the queue is empty,
    //  while begin & end positions are always valid. Begin position is
    //  accessed exclusively be queue reader (front/pop), while back and
    //  end positions are accessed exclusively by queue writer (back/push).
    chunk_t *m_beginChunk;
    INT32    m_beginPos;
    chunk_t *m_backChunk;
    INT32    m_backPos;
    chunk_t *m_endChunk;
    INT32    m_endPos;

    //  People are likely to produce and consume at similar rates.  In
    //  this scenario holding onto the most recently freed chunk saves
    //  us from having to call malloc/free.
    NCAtomicPtr<chunk_t> m_spareChunk;

    // Disable copy construction and assignment.
    NCQueue( const NCQueue & );
    const NCQueue &operator=( const NCQueue & );
};

OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCQUEUE_H_
/* EOF */
