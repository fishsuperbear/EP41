/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: MbufPtr.hpp, reader/topic template need a class with TypeName/Enumerate
 */

#ifndef DDS_CORE_MBUFPTR_HPP
#define DDS_CORE_MBUFPTR_HPP

#include <cstdint>
#include <string>
#include <dp_adapter.h>
#include <RT-DDS/dds/sub/SampleBase.hpp>

namespace dds {
namespace core {
class MbufPtr final : public dds::sub::SampleBase {
public:
    MbufPtr(void) noexcept = default;

    ~MbufPtr(void) final = default;

    void Buffer(Mbuf *buf) noexcept
    {
        mbuf_ = buf;
    }

    Mbuf *Buffer(void) const noexcept
    {
        return mbuf_;
    }

    static std::string TypeName(void)
    {
        return "MbufPtr_RawData";
    }

private:
    Mbuf *mbuf_{nullptr};
};
}
}

#endif /* DDS_CORE_MBUFPTR_HPP */

