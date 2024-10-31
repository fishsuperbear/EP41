/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Tree kv序列化框架
 * Author: s00515168
 */

#ifndef ARA_PER_SERIALIZER_INTERNAL_SERIALIZER_H
#define ARA_PER_SERIALIZER_INTERNAL_SERIALIZER_H

#include <algorithm>
#include "ara/per/serializer/serializer_type.h"
#include "ara/per/per_error_domain.h"

namespace ara {
namespace per {
namespace kv_serialization {

template <typename Container>
class SerializingEnumerator;

template <typename T>
class Serializer
{
public:
    using Type = typename std::decay<T>::type;

    Serializer(const ara::core::String& key, const Type value)
        : key_(key), value_(value)
    {
        static_assert(IsSerializable<Type>(), "This type can not serializable!");
    }

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& kv) const
    {
        SerializeHelper(kv);
    }
private:
    const ara::core::String key_;
    const Type value_;

    template <typename Container, typename U = Type>
    void SerializeHelper(Container& kv, EnableIfBase<U>* = nullptr) const
    {
        kvstype::KvsType element(value_);
        element.SetKey(key_);
        kv = std::move(element);
    }

    template <typename Container, typename U = Type>
    void SerializeHelper(Container& kv, EnableIfEnumerable<U>* = nullptr) const
    {
        SerializingEnumerator<Container> enumerator(key_, kv);
        value_.enumerate_internal(enumerator);
    }
};

template <typename Container>
class SerializingEnumerator
{
public:
    SerializingEnumerator(const ara::core::String& key, Container& containerLocal)
        : kvContainer_(containerLocal)
    {
        kvContainer_.SetKey(key);
    }

    ~SerializingEnumerator() = default;

    template <typename T>
    void operator()(const ara::core::String& key, const T& value)
    {
        kvstype::KvsType element;
        const Serializer<T> serial(key, value);
        serial.Serialize(element);
        (void)kvContainer_.AddKvsArrayItem(element);
    }
private:
    Container& kvContainer_;
};


template <typename First, typename Second>
class Serializer<std::pair<First, Second>>
{
public:
    using Type = std::pair<First, Second>;

    Serializer(const ara::core::String& key, const Type& value)
        : key_(key), value_(value)
    {}

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& kv) const
    {
        kvstype::KvsType firstKvs;
        const Serializer<typename std::decay<First>::type> firstSerializer("key", value_.first);
        firstSerializer.Serialize(firstKvs);
        (void)kv.AddKvsArrayItem(firstKvs);
        kvstype::KvsType secondKvs;
        const Serializer<typename std::decay<Second>::type> secondSerializer("value", value_.second);
        secondSerializer.Serialize(secondKvs);
        (void)kv.AddKvsArrayItem(secondKvs);
        kv.SetKey(key_);
    }
private:
    const ara::core::String key_;
    const Type value_;
};

template <typename T, std::size_t NUM>
class Serializer<ara::core::Array<T, NUM>> {
public:
    using Type = ara::core::Array<T, NUM>;

    Serializer(const ara::core::String& key, const Type& value)
        : key_(key), value_(value)
    {}

    ~Serializer() = default;

    template <typename Container>
    void Serialize(Container& kv) const
    {
        if (NUM == 0) {
            kv.SetKey(key_);
            return;
        }
        SerializeArray(kv);
    }
private:
    const ara::core::String key_;
    const Type value_;

    template <typename Container, typename U = T>
    void SerializeArray(Container& kv, EnableIfBase<U>* = nullptr) const
    {
        kvstype::KvsType element;
        const Serializer<T> serial("", value_[0]);
        serial.Serialize(element);
        for (std::size_t i = 1U; i < NUM; ++i) {
            kvstype::KvsType tmp;
            const Serializer<T> writer("", value_[i]);
            writer.Serialize(tmp);
            (void)element.AddKvsArrayItem(tmp);
        }
        element.SetKey(key_);
        kv = std::move(element);
    }

    template <typename Container, typename U = T>
    void SerializeArray(Container& kv, EnableIfCustom<U>* = nullptr) const
    {
        kvstype::KvsType element;
        for (std::size_t i = 0U; i < NUM; ++i) {
            kvstype::KvsType tmp;
            const Serializer<T> writer(ara::core::String(""), value_[i]);
            writer.Serialize(tmp);
            (void)element.AddKvsArrayItem(tmp);
        }
        element.SetKey(key_);
        kv = std::move(element);
    }
};

template <typename Type>
class ContainerSerializeHelper
{
public:
    ContainerSerializeHelper(const ara::core::String& key, const Type& value)
        : key_(key), value_(value)
    {}

    ~ContainerSerializeHelper() = default;

    template <typename Container>
    void Serialize(Container& kv) const
    {
        kvstype::KvsType element;
        for (const auto& it : value_) {
            kvstype::KvsType tmp;
            const Serializer<typename Type::value_type> writer(ara::core::String(""), it);
            writer.Serialize(tmp);
            (void)element.AddKvsArrayItem(tmp);
        }
        element.SetKey(key_);
        kv = std::move(element);
    }
private:
    const ara::core::String key_;
    const Type value_;
};

template <typename T>
class Serializer<ara::core::Vector<T>>
{
public:
    using Type = ara::core::Vector<T>;
    Serializer(const ara::core::String& key, const Type& value)
        : key_(key), value_(value)
    {}

    ~Serializer() = default;

    template <typename Container, typename U = T>
    void Serialize(Container& kv, EnableIfBase<U>* = nullptr) const
    {
        if (value_.size() == 0) {
            kv.SetKey(key_);
            return;
        }
        auto iter = value_.begin();
        kvstype::KvsType element;
        const auto first_value = *iter;
        const Serializer<T> serial("", first_value);
        serial.Serialize(element);
        iter++;
        while (iter != value_.end()) {
            kvstype::KvsType tmp;
            const auto value = *iter;
            const Serializer<T> writer(ara::core::String(""), value);
            writer.Serialize(tmp);
            (void)element.AddKvsArrayItem(tmp);
            iter++;
        }
        element.SetKey(key_);
        kv = std::move(element);
    }
    template <typename Container, typename U = T>
    void Serialize(Container& kv, EnableIfCustom<U>* = nullptr) const
    {
        if (value_.size() == 0) {
            kv.SetKey(key_);
            return;
        }
        kvstype::KvsType result;
        const ContainerSerializeHelper<ara::core::Vector<T>> serial(key_, value_);
        serial.Serialize(result);
        result.SetKey(key_);
        kv = std::move(result);
    }
private:
    const ara::core::String key_;
    const Type value_;
};

template <typename Key, typename Value>
class Serializer<ara::core::Map<Key, Value>>
    : public ContainerSerializeHelper<ara::core::Map<Key, Value>> {
public:
    using ContainerSerializeHelper<ara::core::Map<Key, Value>>::ContainerSerializeHelper;
    ~Serializer() = default;
};
}  // namespace kv_serialization
}  // namespace per
}  // namespace ara
#endif
