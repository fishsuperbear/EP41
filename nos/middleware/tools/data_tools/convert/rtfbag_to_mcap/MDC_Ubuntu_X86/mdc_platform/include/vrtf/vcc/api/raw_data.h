/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: the implementation of RawData class according to AutoSAR standard core type
 * Create: 2019-07-24
 */
#ifndef ARA_RAWBUFFER_VECTOR_H
#define ARA_RAWBUFFER_VECTOR_H
#include <vector>
namespace vrtf {
namespace core {
template <typename T>
using vector = std::vector<T>;
template <typename T, typename Allocator = std::allocator<T>>
class RawData : public std::vector<T, Allocator> {
public:
    using VectorBase = std::vector<T, Allocator>;
    using VectorBase::VectorBase;
};

template <typename T, typename Allocator>
bool operator == (RawData<T, Allocator> const &lhs,
                  RawData<T, Allocator> const &rhs)
{
    return std::operator == (lhs, rhs);
}

template <typename T, typename Allocator>
bool operator < (RawData<T, Allocator> const &lhs,
                 RawData<T, Allocator> const &rhs)
{
    return std::operator < (lhs, rhs);
}

template <typename T, typename Allocator>
bool operator != (RawData<T, Allocator> const &lhs,
                  RawData<T, Allocator> const &rhs)
{
    return std::operator != (lhs, rhs);
}

template <typename T, typename Allocator>
bool operator > (RawData<T, Allocator> const &lhs,
                 RawData<T, Allocator> const &rhs)
{
    return std::operator > (lhs, rhs);
}

template <typename T, typename Allocator>
bool operator <= (RawData<T, Allocator> const &lhs,
                  RawData<T, Allocator> const &rhs)
{
    return std::operator <= (lhs, rhs);
}

template <typename T, typename Allocator>
void Swap(RawData<T, Allocator> const &lhs, RawData<T, Allocator> const &rhs)
{
    return std::swap(lhs, rhs);
}

template <typename T, typename Allocator>
bool operator >= (RawData<T, Allocator> const &lhs,
                  RawData<T, Allocator> const &rhs)
{
    return std::operator >= (lhs, rhs);
}

using RawBuffer = RawData<std::uint8_t>;
}
}
#endif
