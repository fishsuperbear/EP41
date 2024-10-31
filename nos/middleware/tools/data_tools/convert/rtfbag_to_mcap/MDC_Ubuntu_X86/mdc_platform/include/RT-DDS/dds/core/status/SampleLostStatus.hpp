/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef RT_DDS_SAMPLELOSTSTATUS_HPP
#define RT_DDS_SAMPLELOSTSTATUS_HPP

#include <cstdint>

namespace dds {
namespace sub {
    class DataReaderImpl;
}
}

namespace dds {
namespace core {
namespace status {
class SampleLostStatus {
public:
    /**
     * @ingroup SampleLostStatus
     * @brief return the toltal lost pack count
     * @param[in] None
     * @return uint64_t
     * @req {AR20220610482076}
     */
    uint64_t TotalCount() const noexcept
    {
        return totalCount_;
    }

    /**
     * @ingroup SampleLostStatus
     * @brief return the lost pack count recent time
     * @param[in] None
     * @return uint64_t
     * @req {AR20220610482076}
     */
    uint64_t TotalCountChange() const noexcept
    {
        return totalCountChange_;
    }
private:
    uint64_t totalCount_{0U};
    uint64_t totalCountChange_{0U};
    friend class dds::sub::DataReaderImpl;
};
}
}
}
#endif /* RT_DDS_SAMPLELOSTSTATUS_HPP */
