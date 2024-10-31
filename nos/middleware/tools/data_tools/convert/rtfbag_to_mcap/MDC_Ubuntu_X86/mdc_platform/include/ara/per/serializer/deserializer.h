/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: 数据反序列化
 * Notes: 无
 */
#ifndef ARA_PER_SERIALIZER_DESERIALIZER_H
#define ARA_PER_SERIALIZER_DESERIALIZER_H

#include <algorithm>
#include <securec.h>
#include "ara/per/serializer/serializer_type.h"

namespace ara {
namespace per {
namespace bin_serialization {
template <typename T>
class Deserializer;

class DeserializingEnumerator
{
public:
    DeserializingEnumerator(const std::uint8_t* const data, const std::size_t& size)
        : data_(data), size_(size), pos_(0U)
    {}

    ~DeserializingEnumerator() = default;

    template <typename T>
    void operator()(T& value)
    {
        const Deserializer<T> dataDeserializer(data_ + pos_, size_ - pos_);
        pos_ += dataDeserializer.GetSize();
        if (pos_ > size_) {
            throw std::runtime_error("Deserializing Enumerator failed, Insufficient arguments from remote end.");
        } else {
            value = dataDeserializer.GetValue();
        }
    }
private:
    const std::uint8_t* const data_;
    const std::size_t size_;
    std::size_t pos_;
};

class DeserializerSizeCounter
{
public:
    DeserializerSizeCounter(const std::uint8_t* const data, const std::size_t& dataSize)
        : data_(data), dataSize_(dataSize), size_(0U)
    {}

    ~DeserializerSizeCounter() = default;

    template <typename T>
    void operator()(const T&, std::uint32_t = 0U)
    {
        const Deserializer<T> dataDeserializer(data_ + size_, dataSize_);
        const std::size_t deserializerDataSize = dataDeserializer.GetSize();
        size_ += deserializerDataSize;
        dataSize_ -= deserializerDataSize;
    }

    std::size_t GetSize() const
    {
        return size_;
    }
private:
    const std::uint8_t* const data_;
    std::size_t dataSize_;
    std::size_t size_;
};

template <typename T>
class Deserializer
{
public:
    using value_type = typename std::decay<T>::type;
    using result_type = value_type;

    Deserializer(const std::uint8_t* const data, const std::size_t& size)
        : data_(data), size_(size)
    {
        static_assert(IsSerializable<value_type>(), "Unable to deserialize this type!");
    }

    explicit Deserializer(const Container& v) : Deserializer(v.data(), v.size())
    {}

    ~Deserializer() = default;

    result_type GetValue() const
    {
        return GetValueHelper();
    }

    std::size_t GetSize() const
    {
        return GetSizeHelper();
    }

private:
    const std::uint8_t* const data_;
    const std::size_t size_;

    template <typename U = value_type>
    result_type GetValueHelper(EnableIfScalar<U>* = nullptr) const
    {
        result_type value{};
        if (size_ < sizeof(value_type)) {
            throw std::runtime_error("Deserialization of base type failed, insufficient data.");
        }
        value = *reinterpret_cast<const value_type*>(data_);
        const std::size_t dataSize = sizeof(U);
        switch (dataSize) {
            case 2U: {
                std::uint16_t* const value16 = reinterpret_cast<std::uint16_t*>(&value);
                *value16 = ntohs(*value16);
                break;
            }
            case 4U: {
                std::uint32_t* const value32 = reinterpret_cast<std::uint32_t*>(&value);
                *value32 = ntohl(*value32);
                break;
            }
            case 8U: {
                std::uint64_t* const value64 = reinterpret_cast<std::uint64_t*>(&value);
                *value64 = Ntohl64(*value64);
                break;
            }
            default: break;
        }
        return value;
    }

    template <typename U = value_type>
    result_type GetValueHelper(EnableIfEnumerable<U>* = nullptr) const
    {
        result_type result{};
        DeserializingEnumerator dataDeserializer(data_, size_);
        result.enumerate(dataDeserializer);
        return result;
    }

    template <typename U = value_type>
    std::size_t GetSizeHelper(EnableIfScalar<U>* = nullptr) const
    {
        return sizeof(T);
    }

    template <typename U = value_type>
    std::size_t GetSizeHelper(EnableIfEnumerable<U>* = nullptr) const
    {
        DeserializerSizeCounter counter(data_, size_);
        value_type x;
        x.enumerate(counter);
        return counter.GetSize();
    }
};

template <>
class Deserializer<ara::core::String>
{
public:
    using result_type = ara::core::String;
    using char_type = ara::core::String::value_type;

    Deserializer(const std::uint8_t* const data, const std::size_t& size)
        : data_(data), len_(0U)
    {
        if (size < sizeof(char_type)) {
            throw std::runtime_error("Deserialization of ara::core::String failed, insufficient data.");
        }
        const std::size_t length = *reinterpret_cast<const std::size_t*>(data_);
        len_ = ntohl(length);
    }

    explicit Deserializer(const Container& v) : Deserializer(v.data(), v.size())
    {}

    ~Deserializer() = default;

    result_type GetValue() const
    {
        const char_type* const c = reinterpret_cast<const char_type*>(data_ + sizeof(std::uint32_t));
        return {c, len_};
    }

    std::size_t GetSize() const
    {
        if (len_ == 0U) {
            return (len_) * sizeof(char_type) + sizeof(std::uint32_t);
        }
        return (len_ + 1U) * sizeof(char_type) + sizeof(std::uint32_t);
    }
private:
    const std::uint8_t* const data_;
    std::size_t len_;
};

template <typename Sequence>
class SequenceContainerDeserializer
{
public:
    using result_type = Sequence;

    SequenceContainerDeserializer(const std::uint8_t* const data, const std::size_t& size, const std::size_t& len)
        : data_(data), size_(size), len_(len)
    {}

    ~SequenceContainerDeserializer() = default;

    result_type GetValue() const
    {
        result_type result{};
        const std::uint8_t* current_pos = data_ + sizeof(std::uint32_t);
        std::size_t remaining_size = size_ - sizeof(std::uint32_t);
        std::size_t sizeTmp = len_;
        while (sizeTmp > 0) {
            const Deserializer<typename Sequence::value_type> dataDeserializer(current_pos, remaining_size);
            const std::size_t dataSize = dataDeserializer.GetSize();
            if (dataSize == 0U) {
                break;
            }
            if (remaining_size >= dataSize) {
                if (sizeTmp < dataSize) {
                    throw std::runtime_error("Deserialization of sequence container failed, insufficient data.");
                } else {
                    sizeTmp -= dataSize;
                    current_pos += dataSize;
                    remaining_size -= dataSize;
                    result.push_back(dataDeserializer.GetValue());
                }
            } else {
                throw std::runtime_error("Deserialization of sequence container failed, insufficient remaining data.");
            }
        }
        return std::move(result);
    }

    std::size_t GetSize() const
    {
        std::size_t result{sizeof(std::uint32_t)};
        const std::uint8_t* currentPos = data_ + sizeof(std::uint32_t);
        std::size_t remainingSize = size_ - sizeof(std::uint32_t);
        std::size_t sizeTmp = len_;
        while (sizeTmp > 0) {
            const Deserializer<typename Container::value_type> dataDeserializer(currentPos, remainingSize);
            const std::size_t dataSize = dataDeserializer.GetSize();
            if (remainingSize >= dataSize) {
                if (sizeTmp < dataSize) {
                    throw std::runtime_error("Deserialization of sequence container failed, insufficient data.");
                } else {
                    currentPos += dataSize;
                    remainingSize -= dataSize;
                    result += dataSize;
                    sizeTmp -= dataSize;
                }
            } else {
                throw std::runtime_error("Deserialization of sequence container failed, insufficient remaining data.");
            }
        }
        return std::move(result);
    }
private:
    const std::uint8_t* const data_;
    const std::size_t size_;
    const std::size_t len_;
};

template <>
class Deserializer<ara::core::Vector<bool>>
{
public:
    using result_type = ara::core::Vector<bool>;

    Deserializer(const std::uint8_t* const data, const std::size_t& size)
        : data_(data), size_(size), len_()
    {
        if (size >= sizeof(std::uint32_t)) {
            const std::size_t length = *reinterpret_cast<const std::size_t*>(data_);
            len_ = ntohl(length);
        }
    }

    explicit Deserializer(const Container& v) : Deserializer(v.data(), v.size())
    {}

    ~Deserializer() = default;

    result_type GetValue() const
    {
        return GetValueHelper();
    }

    std::size_t GetSize() const
    {
        return GetSizeHelper();
    }

private:
    const std::uint8_t* const data_;
    const std::size_t size_;
    std::size_t len_;

    result_type GetValueHelper() const
    {
        if (size_ < len_) {
            throw std::runtime_error("GetValue error in deserializer vector , Invalid deserialization.");
        }
        result_type result{};
        std::size_t lenTmp = len_;
        while (lenTmp > 0) {
            bool value{};
            if (lenTmp < sizeof(bool)) {
                break;
            }
            const auto res = memcpy_s(reinterpret_cast<std::uint8_t* >(&value), sizeof(bool),
                data_ + sizeof(std::uint32_t) + len_ - lenTmp, sizeof(bool));
            if (res != 0) {
                throw std::runtime_error("Memory copy error in deserializer vector , Invalid deserialization.");
            }
            switch (sizeof(bool)) {
                case 2U: {
                    std::uint16_t* const value16 = reinterpret_cast<std::uint16_t*>(&value);
                    *value16 = ntohs(*value16);
                    value = static_cast<bool>(*value16);
                    break;
                }
                case 4U: {
                    std::uint32_t* const value32 = reinterpret_cast<std::uint32_t*>(&value);
                    *value32 = ntohl(*value32);
                    value = static_cast<bool>(*value32);
                    break;
                }
                case 8U: {
                    std::uint64_t* const value64 = reinterpret_cast<std::uint64_t*>(&value);
                    *value64 = Ntohl64(*value64);
                    value = static_cast<bool>(*value64);
                    break;
                }
                default: {
                    break;
                }
            }
            lenTmp -= sizeof(bool);
            result.push_back(value);
        }
        return result;
    }

    std::size_t GetSizeHelper(void) const
    {
        return len_ + sizeof(std::uint32_t);
    }
};

template <typename T>
class Deserializer<ara::core::Vector<T>>
{
public:
    using result_type = ara::core::Vector<T>;

    Deserializer(const std::uint8_t* const data, const std::size_t& size)
        : data_(data), size_(size), len_()
    {
        const std::size_t length = *reinterpret_cast<const std::size_t*>(data_);
        len_ = ntohl(length);
    }

    explicit Deserializer(const Container& v) : Deserializer(v.data(), v.size())
    {}

    ~Deserializer() = default;

    result_type GetValue() const
    {
        return GetValueHelper();
    }

    std::size_t GetSize() const
    {
        return GetSizeHelper();
    }

private:
    const std::uint8_t* const data_;
    const std::size_t size_;
    std::size_t len_;

    template <typename U = T>
    result_type GetValueHelper(typename std::enable_if<(!is_enumerable<U>::value) &&
        (std::is_trivially_copyable<U>::value)>::type* = nullptr) const
    {
        const bool isCopyable = ((sizeof(U) == 1) || (!IsLittleEndian()));
        if (isCopyable) {
            result_type result{};
            const std::size_t cpySize = len_;
            result.resize(len_ / sizeof(U));
            
            if (cpySize >= 0) {
                const auto res = memcpy_s(result.data(), cpySize, data_ + sizeof(std::uint32_t), cpySize);
                if ((res != 0) && (cpySize != 0)) {
                    throw std::runtime_error("Memory copy error in deserializer vector , Invalid deserialization.");
                }
            }
            return result;
        }
        return SequenceContainerDeserializer<ara::core::Vector<T>>(data_, size_, len_).GetValue();
    }

    template <typename U = T>
    result_type GetValueHelper(typename std::enable_if<(is_enumerable<U>::value) ||
        (!std::is_trivially_copyable<U>::value)>::type* = nullptr) const
    {
        return SequenceContainerDeserializer<ara::core::Vector<T>>(data_, size_, len_).GetValue();
    }

    template<typename U = T>
    typename std::enable_if<(!is_enumerable<U>::value) && (std::is_trivially_copyable<U>::value), std::size_t>::type
    GetSizeHelper(void) const
    {
        return len_ + sizeof(std::uint32_t);
    }

    template<typename U = T>
    typename std::enable_if<(is_enumerable<U>::value) || (!std::is_trivially_copyable<U>::value), std::size_t>::type
    GetSizeHelper(void) const
    {
        return SequenceContainerDeserializer<ara::core::Vector<T>>(data_, size_, len_).GetSize();
    }
};

template <typename T, std::size_t N>
class Deserializer<ara::core::Array<T, N>> {
public:
    using result_type = ara::core::Array<T, N>;

    Deserializer(const std::uint8_t* const data, const std::size_t& size) : data_(data), size_(size)
    {}

    explicit Deserializer(const Container& v) : Deserializer(v.data(), v.size())
    {}

    ~Deserializer() = default;

    result_type GetValue() const
    {
        return GetArrayValueHelper();
    }

    std::size_t GetSize() const
    {
        return GetArraySizeHelper();
    }

private:
    const std::uint8_t* const data_;
    const std::size_t size_;

    result_type TraverseDeserializeArray() const
    {
        result_type result{};
        std::size_t pos = 0U;
        for (std::size_t i = 0U; i < N; ++i) {
            const Deserializer<T> dataDeserializer(data_ + pos, size_ - pos);
            result[i] = dataDeserializer.GetValue();
            pos += dataDeserializer.GetSize();
        }
        return result;
    }

    template <typename U = T>
    result_type GetArrayValueHelper(typename std::enable_if<(!is_enumerable<U>::value) &&
        (std::is_trivially_copyable<U>::value)>::type* = 0) const
    {
        const bool isCopyable = ((sizeof(U) == 1) || (!IsLittleEndian()));
        if (isCopyable) {
            result_type result{};
            const std::size_t dataSize = N * sizeof(T);
            const auto res = memcpy_s(result.data(), dataSize, data_, dataSize);
            if ((res != 0) && (dataSize != 0)) {
                throw std::runtime_error("Memory copy error in deserializer array , Invalid deserialization.");
            }
            return result;
        }
        return TraverseDeserializeArray();
    }

    template <typename U = T>
    result_type GetArrayValueHelper(typename std::enable_if<(is_enumerable<U>::value) ||
        (!std::is_trivially_copyable<U>::value)>::type* = 0) const
    {
        return TraverseDeserializeArray();
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
        std::size_t pos = 0U;
        for (std::size_t i = 0U; i < N; ++i) {
            const Deserializer<T> dataDeserializer(data_ + pos, size_ - pos);
            pos += dataDeserializer.GetSize();
        }
        return pos;
    }
};

template <typename First, typename Second>
class Deserializer<std::pair<First, Second>>
{
public:
    using result_type = std::pair<First, Second>;

    Deserializer(const std::uint8_t* const data, const std::size_t& size)
        : data_(data), size_(size)
    {}

    explicit Deserializer(const Container& v)
        : Deserializer(v.data(), v.size())
    {}

    ~Deserializer() = default;

    result_type GetValue() const
    {
        const Deserializer<typename std::decay<First>::type> firstDeserializer(data_, size_);
        const std::size_t firstSize = firstDeserializer.GetSize();
        if (firstSize < size_) {
            const Deserializer<typename std::decay<Second>::type> secondDeserializer(data_ + firstSize, size_ - firstSize);
            return result_type(firstDeserializer.GetValue(), secondDeserializer.GetValue());
        } else {
            throw std::runtime_error("Deserialization of pair failed, insufficient data.");
        }
    }

    std::size_t GetSize() const
    {
        const Deserializer<typename std::decay<First>::type> firstDeserializer(data_, size_);
        const std::size_t firstSize = firstDeserializer.GetSize();
        if (firstSize < size_) {
            const Deserializer<typename std::decay<Second>::type> secondDeserializer(data_ + firstSize, size_ - firstSize);
            return firstSize + secondDeserializer.GetSize();
        } else {
            throw std::runtime_error("Deserialization of pair failed, insufficient data.");
        }
    }

private:
    const std::uint8_t* const data_;
    const std::size_t size_;
};

template <typename Container>
class AssociativeDeserializeHelper {
public:
    using result_type = Container;

    AssociativeDeserializeHelper(const std::uint8_t* const data, const std::size_t& size)
        : data_(data), size_(size), len_()
    {
        if (size_ >= sizeof(std::uint32_t)) {
            const std::size_t length = *reinterpret_cast<const std::size_t*>(data_);
            len_ = ntohl(length);
            data_ += sizeof(std::uint32_t);
            size_ -= sizeof(std::uint32_t);
        } else {
            throw std::runtime_error("Deserialization of associative failed, insufficient data.");
        }
    }

    virtual ~AssociativeDeserializeHelper() {}

    result_type GetValue() const
    {
        result_type result{};
        const std::uint8_t* pos = data_;
        std::size_t remaining = size_;

        std::size_t lenTmp = len_;
        while (lenTmp > 0) {
            const Deserializer<typename Container::value_type> dataDeserializer(pos, remaining);
            const std::size_t dataSize = dataDeserializer.GetSize();
            if (lenTmp < dataSize) {
                break;
            }
            lenTmp -= dataSize;
            result.insert(dataDeserializer.GetValue());
            pos += dataSize;
            remaining -= dataSize;
        }
        return result;
    }

    std::size_t GetSize() const
    {
        std::size_t result = sizeof(std::uint32_t);
        const std::uint8_t* pos = data_;
        std::size_t remaining = size_;
        std::size_t lenTmp = len_;
        while (lenTmp > 0) {
            const Deserializer<typename Container::value_type> dataDeserializer(pos, remaining);
            const std::size_t dataSize = dataDeserializer.GetSize();
                if (dataSize > lenTmp) {
                    return 0U;
                }
                lenTmp = lenTmp - dataSize;
                pos += dataSize;
                remaining -= dataSize;
                result += dataSize;
        }
        return result;
    }
private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t len_;
};

template <typename Key, typename Value>
class Deserializer<ara::core::Map<Key, Value>>
    : public AssociativeDeserializeHelper<ara::core::Map<Key, Value>> {
public:
    using AssociativeDeserializeHelper<ara::core::Map<Key, Value>>::AssociativeDeserializeHelper;
    ~Deserializer() = default;
};
}  // namespace bin_serialization
}  // namespace per
}  // namespace ara
#endif
