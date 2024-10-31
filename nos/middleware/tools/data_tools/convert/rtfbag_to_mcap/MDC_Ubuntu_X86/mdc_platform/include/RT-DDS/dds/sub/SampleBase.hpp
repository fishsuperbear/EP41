/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef RT_DDS_SAMPLEBASE_HPP
#define RT_DDS_SAMPLEBASE_HPP


namespace dds {
namespace sub {
/**
 * @brief an interface for the user, so dds can directly deserialize to the given location
 */
class SampleBase {
public:
    virtual ~SampleBase() = default;
};
}
}

#endif // RT_DDS_SAMPLEBASE_HPP
