/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Tree kv反序列化框架
 * Author: s00515168
 */
#ifndef ARA_PER_SERIALIZER_INTERNAL_DESERIALIZER_H
#define ARA_PER_SERIALIZER_INTERNAL_DESERIALIZER_H

#include <algorithm>
#include <securec.h>
#include "ara/per/kvs_type.h"
#include "ara/per/serializer/serializer_type.h"

namespace ara {
namespace per {
namespace kv_serialization {
template <class T, typename std::enable_if<(std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
    std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value), T>::type* = nullptr>
static ara::core::Result<void> GetInternalValue(const kvstype::KvsType& kvstype, T& value)
{
    const auto kvs = kvstype.GetKvsSInt64();
    if (kvs) {
        value = static_cast<T>(std::move(kvs).Value());
        return ara::core::Result<void>::FromValue();
    }
    return ara::core::Result<void>::FromError(kvs.Error());
}

template <class T, typename std::enable_if<(std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value ||
    std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value), T>::type* = nullptr>
static ara::core::Result<void> GetInternalValue(const kvstype::KvsType& kvstype, T& value)
{
    const auto kvs = kvstype.GetKvsUInt64();
    if (kvs) {
        value = static_cast<T>(std::move(kvs).Value());
        return ara::core::Result<void>::FromValue();
    }
    return ara::core::Result<void>::FromError(kvs.Error());
}

template <class T, typename std::enable_if<std::is_same<T, bool>::value, T>::type* = nullptr>
static ara::core::Result<void> GetInternalValue(const kvstype::KvsType& kvstype, T& value)
{
    const auto kvs = kvstype.GetKvsBool();
    if (kvs) {
        value = static_cast<T>(std::move(kvs).Value());
        return ara::core::Result<void>::FromValue();
    }
    return ara::core::Result<void>::FromError(kvs.Error());
}

template <class T, typename std::enable_if<std::is_same<T, ara::core::String>::value, T>::type* = nullptr>
static ara::core::Result<void> GetInternalValue(const kvstype::KvsType& kvstype, T& value)
{
    const auto kvs = kvstype.GetKvsString();
    if (kvs) {
        value = static_cast<T>(std::move(kvs).Value());
        return ara::core::Result<void>::FromValue();
    }
    return ara::core::Result<void>::FromError(kvs.Error());
}

template <class T, typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
static ara::core::Result<void> GetInternalValue(const kvstype::KvsType& kvstype, T& value)
{
    const auto kvs = kvstype.GetKvsDouble();
    if (kvs) {
        value = static_cast<T>(std::move(kvs).Value());
        return ara::core::Result<void>::FromValue();
    }
    return ara::core::Result<void>::FromError(kvs.Error());
}

template <typename T>
class Deserializer;

class DeserializingEnumerator
{
public:
    DeserializingEnumerator(const kvstype::KvsType& kv)
        : kvContainer_(kv)
    {}

    ~DeserializingEnumerator() = default;

    template <typename T>
    void operator()(const ara::core::String& key, T& value, const bool& optional = false) const
    {
        const auto res = kvContainer_.GetKvsArray<ara::per::kvstype::KvsType>();
        if (!res) {
            throw std::runtime_error("Deserializing enumerator, get kvs array failed, key: " + key);
        }
        const auto elements = std::move(res).Value();
        const auto iter = std::find_if(elements.begin(), elements.end(),
            [&key](const kvstype::KvsType& element) { return element.GetKey() == key; });
        if (iter == elements.end()) {
            if (!optional) {
                throw std::runtime_error("Deserializing enumerator, find the target key failed, key: " + key);
            }
            return;
        }
        const kvstype::KvsType kv = std::move(*iter);
        if (kv.GetStatus() != kvstype::KvsType::Status::kSuccess) {
            throw std::runtime_error("Deserializing enumerator, check the target kv failed, key: " + key);
        }
        const Deserializer<T> ds(kv);
        value = ds.GetValue();
    }
private:
    const kvstype::KvsType& kvContainer_;
};

template <typename T>
class Deserializer
{
public:
    using Type = typename std::decay<T>::type;
    explicit Deserializer(const kvstype::KvsType& kv)
        : kvContainer_(kv)
    {
        static_assert(IsSerializable<Type>(), "Unable to deserialize this type!");
    }

    ~Deserializer() = default;

    Type GetValue() const
    {
        return GetValueHelper();
    }
private:
    const kvstype::KvsType& kvContainer_;

    template <typename U = Type>
    Type GetValueHelper(EnableIfBase<U>* = nullptr) const
    {
        Type value{};
        const auto res = GetInternalValue(kvContainer_, value);
        if (!res) {
            const auto key(kvContainer_.GetKey());
            throw std::runtime_error("Deserializing, get internal value failed, key: " + key);
        }
        return value;
    }

    template <typename U = Type>
    Type GetValueHelper(EnableIfEnumerable<U>* = nullptr) const
    {
        Type result{};
        const DeserializingEnumerator deserializerLocal(kvContainer_);
        result.enumerate_internal(deserializerLocal);
        return result;
    }
};

template <typename T>
class SequenceContainerDeserializer
{
public:
    explicit SequenceContainerDeserializer(const kvstype::KvsType& kv)
        : kvContainer_(kv)
    {}

    ~SequenceContainerDeserializer() = default;

    T GetValue() const
    {
        const auto resKvs = kvContainer_.GetKvsArray<kvstype::KvsType>();
        if (!resKvs) {
            const auto key(kvContainer_.GetKey());
            throw std::runtime_error("SequenceContainerDeserializer, get kvs array failed, key: " + key);
        }
        const auto kvs = std::move(resKvs).Value();
        T value{};
        for (const auto& kv : kvs) {
            const Deserializer<typename T::value_type> ds(kv);
            value.push_back(ds.GetValue());
        }
        return std::move(value);
    }
private:
    const kvstype::KvsType& kvContainer_;
};

template <typename T>
class Deserializer<ara::core::Vector<T>>
{
public:
    using Type = ara::core::Vector<T>;
    explicit Deserializer(const kvstype::KvsType& kv)
        : kvContainer_(kv)
    {}

    ~Deserializer() = default;

    Type GetValue() const
    {
        return GetValueHelper();
    }
private:
    const kvstype::KvsType& kvContainer_;

    template <typename U = T>
    Type GetValueHelper(EnableIfBase<U>* = nullptr) const
    {
        const auto res = kvContainer_.GetKvsArray<T>();
        if (!res) {
            const auto key(kvContainer_.GetKey());
            throw std::runtime_error("Deserializing ara::core::Vector<T>, get kvs array failed, key: " + key);
        }
        return std::move(res).Value();
    }
    template <typename U = T>
    Type GetValueHelper(EnableIfCustom<U>* = nullptr) const
    {
        const SequenceContainerDeserializer<ara::core::Vector<T>> deserializerLocal(kvContainer_);
        return deserializerLocal.GetValue();
    }
};

template <typename T, std::size_t NUM>
class Deserializer<ara::core::Array<T, NUM>> {
public:
    using Type = ara::core::Array<T, NUM>;

    explicit Deserializer(const kvstype::KvsType& kv)
        : kvContainer_(kv), key_(kv.GetKey())
    {}

    ~Deserializer() = default;

    Type GetValue() const
    {
        return GetArrayValueHelper();
    }
private:
    const kvstype::KvsType& kvContainer_;
    const ara::core::String key_;
    template <typename U = T>
    Type GetArrayValueHelper(EnableIfBase<U>* = nullptr) const
    {
        const auto res = kvContainer_.GetKvsArray<T>();
        if (!res) {
            throw std::runtime_error("Deserializing ara::core::Array<T, NUM>, get kvs array failed, key: " + key_);
        }
        const auto value = std::move(res).Value();
        if (value.size() != NUM) {
            throw std::runtime_error("Deserializing ara::core::Array<T, NUM>, check array size failed, key: " + key_);
        }
        Type result{};
        for (std::size_t i = 0U; i < NUM; ++i) {
            result[i] = value[i];
        }
        return result;
    }

    template <typename U = T>
    Type GetArrayValueHelper(EnableIfCustom<U>* = nullptr) const
    {
        const auto resKvs = kvContainer_.GetKvsArray<kvstype::KvsType>();
        if (!resKvs) {
            throw std::runtime_error("Deserializer ara::core::Array<T, NUM>, get kvs array failed, key: " + key_);
        }
        const auto kvs = std::move(resKvs).Value();
        if (kvs.size() != NUM) {
            throw std::runtime_error("Deserializing ara::core::Array<T, NUM>, check array size failed, key: " + key_);
        }
        Type result{};
        for (std::size_t i = 0U; i < NUM; ++i) {
            const Deserializer<T> deserializerLocal(kvs[i]);
            result[i] = deserializerLocal.GetValue();
        }
        return std::move(result);
    }
};

template <typename First, typename Second>
class Deserializer<std::pair<First, Second>>
{
public:
    using Type = std::pair<First, Second>;
    explicit Deserializer(const kvstype::KvsType& kv)
        : kvContainer_(kv)
    {}

    ~Deserializer() = default;

    Type GetValue() const
    {
        const auto resKvs = kvContainer_.GetKvsArray<kvstype::KvsType>();
        if (!resKvs) {
            const auto key(kvContainer_.GetKey());
            throw std::runtime_error("Deserializer std::pair<First, Second>, get kvs array failed, key: " + key);
        }
        const auto kvs = std::move(resKvs).Value();
        if (kvs.size() != 2U) {
            const auto key(kvContainer_.GetKey());
            throw std::runtime_error("Deserializing std::pair<First, Second>, check array size failed, key: " + key);
        }
        const Deserializer<typename std::decay<First>::type> firstDeserializer(kvs[0]);
        const auto firstValue = firstDeserializer.GetValue();
        const Deserializer<typename std::decay<Second>::type> secondDeserializer(kvs[1]);
        const auto secondValue = secondDeserializer.GetValue();
        return std::make_pair(firstValue, secondValue);
    }
private:
    const kvstype::KvsType& kvContainer_;
};

template <typename T>
class AssociativeDeserializeHelper
{
public:
    explicit AssociativeDeserializeHelper(const kvstype::KvsType& kv)
        : kvContainer_(kv)
    {}

    ~AssociativeDeserializeHelper() = default;

    T GetValue() const
    {
        const auto resKvs = kvContainer_.GetKvsArray<kvstype::KvsType>();
        if (!resKvs) {
            const auto key(kvContainer_.GetKey());
            throw std::runtime_error("AssociativeDeserializeHelper, get kvs array failed, key: " + key);
        }
        const auto kvs = std::move(resKvs).Value();
        T value{};
        for (const auto& kv : kvs) {
            const Deserializer<typename T::value_type> ds(kv);
            value.insert(ds.GetValue());
        }
        return std::move(value);
    }
private:
    const kvstype::KvsType& kvContainer_;
};

template <typename Key, typename Value>
class Deserializer<ara::core::Map<Key, Value>>
    : public AssociativeDeserializeHelper<ara::core::Map<Key, Value>> {
public:
    using AssociativeDeserializeHelper<ara::core::Map<Key, Value>>::AssociativeDeserializeHelper;
    ~Deserializer() = default;
};
}  // namespace kv_serialization
}  // namespace per
}  // namespace ara
#endif
