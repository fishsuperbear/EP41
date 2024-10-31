/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Sample.hpp
 */

#ifndef DDS_SUB_SAMPLE_HPP
#define DDS_SUB_SAMPLE_HPP

#include <vector>
#include <RT-DDS/dds/sub/SampleInfo.hpp>

namespace dds {
namespace sub {
/**
 * @brief This class encapsulate the data and SampleInfo associated with DDS samples.
 */
template<typename T>
class Sample {
public:
    /**
     * @brief Get the data of type T associated with this Sample.
     * @return const reference of type T.
     */
    const T &Data() const noexcept
    {
        return data_;
    }

    void Data(T value) noexcept
    {
        data_ = std::move(value);
    }

    /**
     * @brief Get the SampleInfo associated with this Sample.
     * @return const reference of SampleInfo.
     */
    const SampleInfo &Info() const noexcept
    {
        return info_;
    }

    void Info(SampleInfo value) noexcept
    {
        info_ = std::move(value);
    }

private:
    T data_{};
    SampleInfo info_{};

    template<typename TT>
    friend class DataReader;
};

extern const int32_t SAMPLE_TAKE_NUM_MAX; /// 1000

/**
 * @brief Sequence of Sample of type T.
 */
template<typename T>
using SampleSeq = std::vector<Sample<T>>;
}
}

#endif /* DDS_SUB_SAMPLE_HPP */

