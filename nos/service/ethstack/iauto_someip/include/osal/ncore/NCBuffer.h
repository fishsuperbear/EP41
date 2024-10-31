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
 * @file NCBuffer.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef INCLUDE_NCORE_NCBUFFER_H_
#define INCLUDE_NCORE_NCBUFFER_H_

#include <cstdint>
#include <ostream>
#include <vector>

#include "osal/ncore/NCNameSpace.h"

OSAL_BEGIN_NAMESPACE

using Buffer = std::vector<uint8_t>;

/// @brief Represents a view over data held in another container
class BufferView {
   public:
    BufferView() = delete;

    /**
     * @brief Constructs a buffer view from a byte sequence represented as a pointer to the data
     *        and its length
     *
     * @param dataPtr     pointer to data
     * @param dataLength  number of elements in dataPtr
     */
    BufferView( const uint8_t* dataPtr, size_t dataLength );

    /**
     * @brief Constructs a buffer view from std::vector of bytes.
     *
     * @param buf vector of bytes
     */
    explicit BufferView( const Buffer& buf ) noexcept
        : dataPtr_{ buf.data() }, dataLength_{ buf.size() } {}

    /**
     * @brief Constructs a buffer view from the part of a vector of bytes starting from the
     *        beginning.
     *
     * @param buf  vector of bytes
     * @param length length of the buffer in bytes
     */
    BufferView( const Buffer& buf, size_t length );

    /**
     * @brief Constructs a buffer view from part of a vector set by begin and end positions.
     *
     * @param buf ector of bytes
     * @param begin start position of a view
     * @param end end position of a view
     */
    BufferView( const Buffer& buf, size_t begin, size_t end );

    /**
     * @brief Defaulted copy constructor
     *
     * @param oth another BufferView
     */
    BufferView( const BufferView& oth ) = default;

    /**
     * @brief Constructs a buffer view from other buffer view by moving pointer to data and length
     *
     * @param oth another buffer view
     */
    BufferView( BufferView&& oth ) noexcept;

    /**
     * @brief Defaulted assignment operator
     *
     * @param oth nother buffer view
     * @return BufferView& new BUfferView
     */
    BufferView& operator=( const BufferView& oth ) = default;

    /**
     * @brief move-assign buffer view from antoher buffer view
     *
     * @param oth another buffer view
     * @return BufferView& new BufferView
     */
    BufferView& operator=( BufferView&& oth ) noexcept;

    /**
     * @brief Destroy the Buffer View object
     *
     */
    ~BufferView() noexcept = default;

    /**
     * @brief Returns starting position of a view
     *
     * @return const uint8_t* pointer to the beginning of the buffer
     */
    const uint8_t* begin() const noexcept { return dataPtr_; }

    /**
     * @brief Returns end position of a view
     *
     * @return const uint8_t* pointer to the end of the buffer

     */
    const uint8_t* end() const noexcept { return dataPtr_ + dataLength_; }

    /**
     * @brief Check if buffer view points to the data
     *
     * @return true if buffer view represent real data buffer
     * @return false othervise
     */
    bool isValid() const noexcept;

   private:
    /// @brief   pointer to data represented by a buffer view
    const uint8_t* dataPtr_;

    /// @brief   length of the data represented by a buffer view.
    size_t dataLength_;
};

OSAL_END_NAMESPACE

/**
 * @brief Represent buffer content as a HEX-string suitable for printing
 *
 * @param os output stream
 * @param buffer buffer to be represented as a sequence of hex values
 * @return std::ostream& output stream
 */
std::ostream& operator<<( std::ostream& os, const OSAL::Buffer& buffer );

#endif  // INCLUDE_NCORE_NCBUFFER_H_

/* EOF */