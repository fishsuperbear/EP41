/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: 数据序列化
 * Notes: 无
 */
#ifndef ARA_PER_SERIALIZER_SERIALIZER_H
#define ARA_PER_SERIALIZER_SERIALIZER_H

#include <algorithm>
#include <securec.h>
#include "ara/per/serializer/serializer_type.h"

namespace ara {
namespace per {
namespace bin_serialization {

template <typename Container>
class SerializingEnumerator;

template <typename T>
class Serializer;

class SerializeSizeCounter {
public:
    SerializeSizeCounter() :size_(0U)
    {}

    ~SerializeSizeCounter() = default;

    template <typename T>
    void operator()(const T& value)
    {
        const Serializer<T> dataSerializer(value);
        size_ += dataSerializer.GetSize();
    }

    std::size_t GetSize() const
    {
        return size_;
    }

private:
    std::size_t size_;
};

template <typename T>
class Serializer
{
public:
    using value_type = typename std::decay<T>::type;

    Serializer(const value_type& value) : value_(value)
    {
        static_assert(IsSerializable<value_type>(), "This type can not serializable!");
    }

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& c) const
    {
        SerializeHelper(c);
    }

    std::size_t GetSize() const
    {
        return GetSizeHelper();
    }

private:
    const value_type& value_;

    template <typename Container, typename U = value_type>
    void SerializeHelper(Container& c, EnableIfScalar<U>* = nullptr) const
    {
        const std::size_t dataSize = sizeof(T);
        value_type v;
        v = value_;
        switch (dataSize) {
            case 2U: {
                std::uint16_t* const value16 = reinterpret_cast<std::uint16_t*>(&v);
                *value16 = htons(*value16);
                break;
            }
            case 4U: {
                std::uint32_t* const value32 = reinterpret_cast<std::uint32_t*>(&v);
                *value32 = htonl(*value32);
                break;
            }
            case 8U: {
                std::uint64_t* const value64 = reinterpret_cast<std::uint64_t*>(&v);
                *value64 = Htonl64(*value64);
                break;
            }
            default: break;
        }
        const std::uint8_t* const data = reinterpret_cast<const std::uint8_t*>(&v);
        (void)std::copy(data, data + sizeof(value_), std::back_inserter(c));
    }

    template <typename Container, typename U = value_type>
    void SerializeHelper(Container& c, EnableIfEnumerable<U>* = nullptr) const
    {
        SerializingEnumerator<Container> enumerator(c);
        const_cast<value_type&>(value_).enumerate(enumerator);
    }

    template <typename U = value_type>
    typename std::enable_if<is_enumerable<U>::value, std::size_t>::type
    GetSizeHelper() const
    {
        SerializeSizeCounter sizeCounter;
        (const_cast<value_type&>(value_)).enumerate(sizeCounter);
        return sizeCounter.GetSize();
    }

    template <typename U = value_type>
    typename std::enable_if<(!is_enumerable<U>::value) && (std::is_trivially_copyable<U>::value), std::size_t>::type
    GetSizeHelper() const
    {
        return sizeof(U);
    }
};

template <typename Container>
class SerializingEnumerator
{
public:
    SerializingEnumerator(Container& containerLocal)
        : dataContainer_(containerLocal), size_(0U)
    {}

    ~SerializingEnumerator() = default;

    template <typename T>
    void operator()(const T& value, std::uint32_t = 0U)
    {
        const Serializer<T> dataSerializer(value);
        dataSerializer.Serialize(dataContainer_);
        size_ += dataSerializer.GetSize();
    }

    std::size_t GetSize() const
    {
        return size_;
    }

private:
    Container& dataContainer_;
    std::size_t size_;
};

template <>
class Serializer<ara::core::String>
{
public:
    using value_type = ara::core::String;
    
    Serializer(const value_type& value) : str_(value)
    {}

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& c) const
    {
        const std::size_t len = htonl(str_.size());
        const std::uint8_t* const lenData = reinterpret_cast<const std::uint8_t*>(&len);
        (void)std::copy(lenData, lenData + sizeof(std::uint32_t), std::back_inserter(c));
        if (len == 0U) {
            return;
        }
        const typename Container::value_type* const begin =
            reinterpret_cast<const typename Container::value_type*>(str_.c_str());
        const typename Container::value_type* const end = begin + str_.size() + sizeof('\0');
        (void)std::copy(begin, end, std::back_inserter(c));
    }

    std::size_t GetSize() const
    {
        return sizeof(std::uint32_t) + str_.size() + sizeof('\0');
    }
private:
    const ara::core::String& str_;
};

template <typename First, typename Second>
class Serializer<std::pair<First, Second>>
{
public:
    using value_type = std::pair<First, Second>;

    Serializer(const value_type& value) : value_(value)
    {}

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& c) const
    {
        const Serializer<typename std::decay<First>::type> first_serializer(value_.first);
        const Serializer<typename std::decay<Second>::type> second_serializer(value_.second);
        first_serializer.Serialize(c);
        second_serializer.Serialize(c);
    }

    std::size_t GetSize() const
    {
        const Serializer<typename std::decay<First>::type> firstSerializer(value_.first);
        const Serializer<typename std::decay<Second>::type> secondSerializer(value_.second);
        return (firstSerializer.GetSize() + secondSerializer.GetSize());
    }
private:
    const value_type& value_;
};

template <typename T, std::size_t N>
class Serializer<ara::core::Array<T, N>> {
public:
    using value_type = ara::core::Array<T, N>;

    Serializer(const value_type& value) : value_(value)
    {}

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& c) const
    {
        SerializeArrayHelper(c);
    }

    std::size_t GetSize() const
    {
        return GetArraySizeHelper();
    }

private:
    const value_type&  value_;

    template <typename Container>
    void TraverseSerializeArray(Container& c) const
    {
        for (std::size_t i = 0U; i < N; ++i) {
            const Serializer<T> dataSerializer(value_[i]);
            dataSerializer.Serialize(c);
        }
    }

    template <typename U = T>
    void SerializeArrayHelper(std::vector<uint8_t> &c, typename std::enable_if<(!is_enumerable<U>::value) &&
        (std::is_trivially_copyable<U>::value)>::type* = nullptr) const
    {
        const bool isCopyable = ((sizeof(U) == 1) || (!IsLittleEndian()));
        if (isCopyable) {
            const std::size_t len = sizeof(T) * N;
            if (len == 0U) {
                return;
            }
            const std::uint8_t* const data = reinterpret_cast<const std::uint8_t*>(&value_);
            (void)std::copy(data, data + len, std::back_inserter(c));
            return;
        }
        TraverseSerializeArray(c);
    }

    template <typename U = T>
    void SerializeArrayHelper(std::vector<uint8_t> &c, typename std::enable_if<(is_enumerable<U>::value) ||
       (!std::is_trivially_copyable<U>::value)>::type* = nullptr) const
    {
        TraverseSerializeArray(c);
    }

    template<typename U = T>
    typename std::enable_if<(!is_enumerable<U>::value) && (std::is_trivially_copyable<U>::value), std::size_t>::type
    GetArraySizeHelper() const
    {
        return sizeof(T) * N;
    }

    template<typename U = T>
    typename std::enable_if<(is_enumerable<U>::value) || (!std::is_trivially_copyable<U>::value), std::size_t>::type
    GetArraySizeHelper() const
    {
        std::size_t totalSize = 0U;
        for (const T& item : value_) {
            const Serializer<T> dataSerializer(item);
            totalSize += dataSerializer.GetSize();
        }
        return totalSize;
    }
};

template <typename Container>
class ContainerSerializeHelper
{
public:
    using value_type = Container;

    ContainerSerializeHelper(const Container& containerLocal) : dataContainer_(containerLocal)
    {}

    ~ContainerSerializeHelper() = default;

    template <typename TargetContainer>
    void Serialize(TargetContainer& c) const
    {
        std::size_t len = GetSize() - sizeof(std::uint32_t);
        len = htonl(len);
        const std::uint8_t* const lenData = reinterpret_cast<const std::uint8_t*>(&len);
        (void)std::copy(lenData, lenData + sizeof(std::uint32_t), std::back_inserter(c));
        for (const typename Container::value_type& item : dataContainer_) {
            const Serializer<typename Container::value_type> dataSerializer(item);
            dataSerializer.Serialize(c);
        }
    }

    std::size_t GetSize() const
    {
        std::size_t dataSize = sizeof(std::uint32_t);
        for (const typename Container::value_type& item : dataContainer_) {
            const Serializer<typename Container::value_type> dataSerializer(item);
            dataSize += dataSerializer.GetSize();
        }
        return dataSize;
    }
private:
    const Container& dataContainer_;
};

template <>
class Serializer<ara::core::Vector<bool>>
{
public:
    using value_type = ara::core::Vector<bool>;

    Serializer(const value_type& value) : dataContainer_(value)
    {}

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& c) const
    {
        SerializeHelper(c);
    }
    std::size_t GetSize() const
    {
        return GetSizeHelper();
    }
private:
    const value_type& dataContainer_;

    template <typename Container>
    void SerializeHelper(Container& c) const
    {
        const std::size_t len = GetSize() - sizeof(std::uint32_t);
        if (len == 0U) {
            return;
        }
        const std::size_t tmp = htonl(len);
        const std::uint8_t* const lenData = reinterpret_cast<const std::uint8_t*>(&tmp);
        (void)std::copy(lenData, lenData + sizeof(std::uint32_t), std::back_inserter(c));
        const std::size_t s {sizeof(bool)};
        for (std::size_t i{0U}; i < len; i++) {
            bool v = dataContainer_[i];
            const std::uint8_t* const point = reinterpret_cast<std::uint8_t*>(&v);
            switch (s) {
                case 2U: {
                    v = static_cast<bool>(htons(static_cast<std::uint16_t>(v)));
                    break;
                } 
                case 4U: {
                    v = static_cast<bool>(htonl(static_cast<std::uint32_t>(v)));
                    break;
                }
                case 8U: {
                    v = static_cast<bool>(Htonl64(static_cast<std::uint64_t>(v)));
                    break;
                }
                default: {
                    break;
                }
            }
            (void)std::copy(point, point + s, std::back_inserter(c));
        }
        return;
    }

    std::size_t GetSizeHelper() const
    {
        return sizeof(bool) * dataContainer_.size() + sizeof(std::uint32_t);
    }
};

template <typename T>
class Serializer<ara::core::Vector<T>>
{
public:
    using value_type = ara::core::Vector<T>;

    Serializer(const value_type& value) : dataContainer_(value)
    {}

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& c) const
    {
        SerializeHelper(c);
    }
    std::size_t GetSize() const
    {
        return GetSizeHelper();
    }
private:
    const value_type& dataContainer_;

    template <typename Container, typename U = T>
    void SerializeHelper(Container& c, typename std::enable_if<(!is_enumerable<U>::value) &&
        (std::is_trivially_copyable<U>::value)>::type* = nullptr) const
    {
        const bool isCopyable = ((sizeof(T) == 1) || (!IsLittleEndian()));
        if (isCopyable) {
            std::size_t len = GetSize() - sizeof(std::uint32_t);
            len = htonl(len);
            const std::uint8_t* const lenData = reinterpret_cast<const std::uint8_t*>(&len);
            (void)std::copy(lenData, lenData + sizeof(std::uint32_t), std::back_inserter(c));
            const std::size_t dataSize = GetSize() - sizeof(std::uint32_t);
            if (dataSize == 0U) {
                return;
            }
            const std::uint8_t* const valueData = reinterpret_cast<const std::uint8_t*>(dataContainer_.data());
            (void)std::copy(valueData, valueData + dataSize, std::back_inserter(c));
            return;
        }
        const ContainerSerializeHelper<ara::core::Vector<T>> dataSerializer(dataContainer_);
        dataSerializer.Serialize(c);
    }

    template <typename Container, typename U = T>
    void SerializeHelper(Container& c, typename std::enable_if<(is_enumerable<U>::value) ||
        (!std::is_trivially_copyable<U>::value)>::type* = nullptr) const
    {
        const ContainerSerializeHelper<ara::core::Vector<T>> dataSerializer(dataContainer_);
        dataSerializer.Serialize(c);
    }

    template<typename U = T>
    typename std::enable_if<(!is_enumerable<U>::value) && (std::is_trivially_copyable<U>::value), std::size_t>::type
    GetSizeHelper() const
    {
        const std::size_t dataSize = sizeof(T) * dataContainer_.size() + sizeof(std::uint32_t);
        return dataSize;
    }

    template<typename U = T>
    typename std::enable_if<(is_enumerable<U>::value) || (!std::is_trivially_copyable<U>::value), std::size_t>::type
    GetSizeHelper() const
    {
        std::size_t dataSize = 0U;
        for (const T& item : dataContainer_) {
            const Serializer<T> dataSerializer(item);
            dataSize += dataSerializer.GetSize();
        }
        dataSize += sizeof(std::uint32_t);
        return dataSize;
    }
};

template <typename Key, typename Value>
class Serializer<ara::core::Map<Key, Value>>
    : public ContainerSerializeHelper<ara::core::Map<Key, Value>> {
public:
    using ContainerSerializeHelper<ara::core::Map<Key, Value>>::ContainerSerializeHelper;
    ~Serializer() = default;
};
}  // namespace bin_serialization
}  // namespace per
}  // namespace ara
#endif
