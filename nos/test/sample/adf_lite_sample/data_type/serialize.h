#pragma once
#include <map>

//基本类型的序列化，如int32_t, int64_t
#define SERIALIZE_DATA(data, serialize_str, offset) \
    { copy((char*)&data, (char*)&data + sizeof(data), serialize_str.begin() + offset); \
    offset += sizeof(data); }

//基本类型的序列化，不修改offset值
#define SERIALIZE_DATA_PURE(data, serialize_str, offset) \
    {   copy((char*)&data, (char*)&data + sizeof(data), serialize_str.begin() + offset); }

//基本类型的反序列化
#define UNSERIALIZE_DATA(type, target, serialize_str, offset) \
    {target = *(type*)(serialize_str.data() + offset); \
    offset += sizeof(type); }

//string类型的序列化
#define SERIALIZE_STRING(str, serialize_str, offset) \
    {   uint32_t size = str.size(); \
        SERIALIZE_DATA(size, serialize_str, offset) \
    copy(str.begin(), str.end(),  serialize_str.begin() + offset); \
    offset += str.size(); \
    }

// 从serialize_str的offset位置反序列化后赋值给target变量(string类型)
#define UNSERIALIZE_STRING(target, serialize_str, offset) \
    { uint32_t size = *(uint32_t*)(serialize_str.data() + offset); \
    offset += sizeof(uint32_t); \
    target = serialize_str.substr(offset, size); \
    offset += size; \
    }

// 将vec（vector类型)序列化到serialize_str的offset位置
#define SERIALIZE_VECTOR(vec, serialize_str, offset) \
    { uint32_t size = vec.size(); \
    SERIALIZE_DATA(size, serialize_str, offset) \
    for (uint32_t i = 0; i < size; i++) { \
        SERIALIZE_DATA(vec[i], serialize_str, offset) \
    } \
    }

// mem_type: 成员类型   从serialize_str的offset位置反序列化后赋值给target变量(string类型)
#define UNSERIALIZE_VECTOR(target, mem_type, serialize_str, offset) \
    { uint32_t size; \
    UNSERIALIZE_DATA(uint32_t, size, serialize_str, offset) \
    for (uint32_t i = 0; i < size; i++) { \
        mem_type tmp;   \
        UNSERIALIZE_DATA(mem_type, tmp, serialize_str, offset) \
        target.push_back(tmp); \
    } \
    }

template <typename T>
void serialize_data(T& data, std::string& serialize_str, uint32_t& offset) {
    copy((char*)&data, (char*)&data + sizeof(data), serialize_str.begin() + offset);
    offset += sizeof(data);
}

template <typename T>
void serialize_data_pure(T& data, std::string& serialize_str, uint32_t& offset) {
    copy((char*)&data, (char*)&data + sizeof(data), serialize_str.begin() + offset);
}

template <typename T>
void unserialize_data(T& target, std::string& serialize_str, uint32_t& offset) {
    target = *(T*)(serialize_str.data() + offset);
    offset += sizeof(T);
}

void serialize_string(std::string& str, std::string& serialize_str, uint32_t& offset) {
    uint32_t size = str.size();
    serialize_data(size, serialize_str, offset);
    copy(str.begin(), str.end(),  serialize_str.begin() + offset);
    offset += str.size();
}

void unserialize_string(std::string& target, std::string& serialize_str, uint32_t& offset) {
    uint32_t size = *(uint32_t*)(serialize_str.data() + offset);
    offset += sizeof(uint32_t);
    target = serialize_str.substr(offset, size);
    offset += size;
}

template <typename T>
void serialize_vector(std::vector<T>& vec, std::string& serialize_str, uint32_t& offset) {
    uint32_t size = vec.size();
    serialize_data(size, serialize_str, offset);
    for (uint32_t i = 0; i < size; i++) {
        serialize_data(vec[i], serialize_str, offset);
    }
}

template <typename T>
void unserialize_vector(std::vector<T>& target, std::string& serialize_str, uint32_t& offset) {
    uint32_t size;
    unserialize_data<uint32_t>(size, serialize_str, offset);
    for (uint32_t i = 0; i < size; i++) {
        T tmp;
        unserialize_data<T>(tmp, serialize_str, offset);
        target.push_back(tmp);
    }
}
