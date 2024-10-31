/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#ifndef E2E_INCLUDE_CRC_BUFFER_H_
#define E2E_INCLUDE_CRC_BUFFER_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include <ostream>
#include <vector>

namespace crc {

using Buffer = std::vector<uint8_t>;

/// @brief Represents a view over data held in another container
class BufferView {
   public:
    BufferView() = delete;

    /// @brief Constructs a buffer view from a byte sequence represented as a pointer to the data
    /// and its length
    ///
    /// @param dataPtr     pointer to data
    /// @param dataLength  number of elements in @a dataPtr
    ///
    /// @throws std::invalid_argument is thrown if pointer to data is null or data length is zero
    BufferView( const uint8_t* dataPtr, size_t dataLength );

    /// @brief Constructs a buffer view from std::vector of bytes.
    ///
    /// @param buf    vector of bytes
    explicit BufferView( const Buffer& buf ) noexcept
        : dataPtr_{buf.data()}, dataLength_{buf.size()} {}

    /// @brief Constructs a buffer view from the part of a vector of bytes starting from the
    /// beginning.
    ///
    /// @param buf  vector of bytes
    /// @param length  length of the buffer in bytes
    ///
    /// @throws std::length_error is thrown if given length is equal to zero or exceeds buf size
    BufferView( const Buffer& buf, size_t length );

    /// @brief Constructs a buffer view from part of a vector set by begin and end positions.
    ///
    /// @param buf  vector of bytes
    /// @param begin start position of a view
    /// @param end   end position of a view
    ///
    /// @throws std::invalid_argument is thrown if begin position is greater than end position
    /// @throws std::out_of_range is thrown if end position is greater than buf size
    BufferView( const Buffer& buf, size_t begin, size_t end );

    /// @brief Defaulted copy constructor
    BufferView( const BufferView& oth ) = default;

    /// @brief Constructs a buffer view from other buffer view by moving pointer to data and length
    ///
    /// @param oth    another buffer view
    BufferView( BufferView&& oth ) noexcept;

    /// @brief Defaulted assignment operator
    BufferView& operator=( const BufferView& oth ) = default;

    /// @brief move-assign buffer view from antoher buffer view
    ///
    /// @param oth    another buffer view
    BufferView& operator=( BufferView&& oth ) noexcept;

    ~BufferView() noexcept = default;

    /// @brief Returns starting position of a view
    ///
    /// @return  pointer to the beginning of the buffer
    const uint8_t* begin() const noexcept { return dataPtr_; }

    /// @brief Returns end position of a view
    ///
    /// @return  pointer to the end of the buffer
    const uint8_t* end() const noexcept { return dataPtr_ + dataLength_; }

    /// @brief Check if buffer view points to the data
    ///
    /// @return  true if buffer view represent real data buffer, false - othervise
    bool isValid() const noexcept;

   private:
    /// @brief   pointer to data represented by a buffer view
    const uint8_t* dataPtr_;

    /// @brief   length of the data represented by a buffer view.
    size_t dataLength_;
};

}  // namespace crc

namespace std {
/// @brief Represent buffer content as a HEX-string suitable for printing
///
/// @param os output stream
/// @param buffer   buffer to be represented as a sequence of hex values
///
/// @return output stream
std::ostream& operator<<( std::ostream& os, const crc::Buffer& buffer );

}

#endif  // E2E_INCLUDE_CRC_BUFFER_H_
        /* EOF */