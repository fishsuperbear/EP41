/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef RT_DDS_SAMPLETIMEOUTSTATUS_HPP
#define RT_DDS_SAMPLETIMEOUTSTATUS_HPP

#include <cstdint>
#include <string>

namespace dds {
namespace sub {
class DataReaderImpl;
}
}

namespace dds {
namespace core {
namespace status {

class SampleTimeOutStatus {
public:
    /**
      * @ingroup SampleTimeOutStatus
      * @brief return the PlogMSGUid of the Recently timeout pack
      * @param[in] None
      * @return uint64_t
      */
    uint64_t PlogMSGUid() const noexcept
    {
        return PlogMSGUid_;
    }

    /**
     * @ingroup SampleTimeOutStatus
     * @brief return the Pid of writer
     * @param[in] None
     * @return uint32_t
     */
    uint32_t WriterPid() const noexcept
    {
        return writerPid_;
    }

    /**
     * @ingroup SampleTimeOutStatus
     * @brief return the Pid of reader
     * @param[in] None
     * @return uint32_t
     */
    uint32_t ReaderPid() const noexcept
    {
        return readerPid_;
    }

    /**
      * @ingroup SampleTimeOutStatus
      * @brief return the total count of timeout packs
      * @param[in] None
      * @return uint64_t
      */
    uint64_t TotalCount() const noexcept
    {
        return totalCount_;
    }

    /**
      * @ingroup SampleTimeOutStatus
      * @brief return the count of timeout packs recent time
      * @param[in] None
      * @return uint64_t
      */
    uint64_t RecentCount() const noexcept
    {
        return recentCount_;
    }

    /**
      * @ingroup SampleTimeOutStatus
      * @brief return the GuidStr
      * @param[in] None
      * @return std::string
      */
    std::string GuidStr() const noexcept
    {
        return guidStr_;
    }

private:
    uint64_t PlogMSGUid_{0U};
    uint64_t totalCount_{0U};
    uint64_t recentCount_{0U};
    uint32_t writerPid_{0U};
    uint32_t readerPid_{0U};
    std::string guidStr_{};

    friend class dds::sub::DataReaderImpl;
};
}
}
}

#endif // RT_DDS_SAMPLETIMEOUTSTATUS_HPP
