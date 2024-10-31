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
#include <cctype>
#include <iomanip>
#include <stdexcept>
#include "extend/crc/buffer.h"

namespace crc {

BufferView::BufferView( const uint8_t* ptr, size_t length ) : dataPtr_{ptr}, dataLength_{length} {
    if ( !isValid() ) {
        throw std::invalid_argument{"data is incorrect"};
    }
}

BufferView::BufferView( const Buffer& buf, size_t length )
    : dataPtr_{buf.data()}, dataLength_{length} {
    if ( ( length > buf.size() ) || ( length == 0U ) ) {
        throw std::length_error{"Length is incorrect."};
    }
}

BufferView::BufferView( const Buffer& buf, size_t begin, size_t end )
    : dataPtr_{buf.data() + begin}, dataLength_( end - begin ) {
    if ( begin > end ) {
        throw std::invalid_argument{"Incorrect buffer boundaries (begin > end)."};
    }
    if ( end > buf.size() ) {
        throw std::out_of_range{"Length is incorrect (end > buf.size())."};
    }
}

BufferView::BufferView( BufferView&& oth ) noexcept
    : dataPtr_{oth.dataPtr_}, dataLength_{oth.dataLength_} {
    oth.dataPtr_    = nullptr;
    oth.dataLength_ = 0U;
}

BufferView& BufferView::operator=( BufferView&& oth ) noexcept {
    if ( this != &oth ) {
        dataPtr_        = oth.dataPtr_;
        dataLength_     = oth.dataLength_;
        oth.dataPtr_    = nullptr;
        oth.dataLength_ = 0U;
    }
    return *this;
}

bool BufferView::isValid() const noexcept {
    if ( dataLength_ == 0U ) {
        return true;
    }

    return dataPtr_ != nullptr;
}

}  // namespace crc

namespace std {
std::ostream& operator<<( std::ostream& os, const crc::Buffer& buffer ) {
    for ( auto&& byte : buffer ) {
        if ( std::isupper( byte ) != 0 ) {
            os << byte;
        } else {
            os << "[" << std::setfill( '0' ) << std::setw( 2 ) << std::hex
               << static_cast<uint32_t>( byte ) << std::dec << "]";
        }
    }
    return os;
}
}
