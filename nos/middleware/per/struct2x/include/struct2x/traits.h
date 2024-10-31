/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: traits
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_STRUCT2X_INCLUDE_STRUCT2X_TRAITS_H_
#define MIDDLEWARE_PER_STRUCT2X_INCLUDE_STRUCT2X_TRAITS_H_

#include <stdint.h>
#include <stdlib.h>

#include <string>
#include <vector>

#ifdef _MSC_VER_  // for MSVC
#define FORCEINLINE __forceinline
#elif defined __GNUC__  // for gcc on Linux/Apple OS X
#define FORCEINLINE __inline__ __attribute__((always_inline))
#else
#define FORCEINLINE
#endif

#ifdef _MSC_VER
#ifdef EXPORTAPI
#define EXPORTAPI _declspec(dllimport)
#else
#define EXPORTAPI _declspec(dllexport)
#endif
#else
#define EXPORTAPI
#endif

namespace struct2x {

const uint32_t BITNUM = 16;

enum {
    TYPE_VARINT = 0,   // int32,int64,uint32,uint64,bool,enum
    TYPE_SVARINT = 1,  // sint32,sin64
    TYPE_FIXED32 = 2,  // fixed32,sfixed32
    TYPE_FIXED64 = 3,  // fixed64,sfixed64
    TYPE_BYTES = 4,    // bytes
    TYPE_PACK = 5,     // repaeted [pack=true]
};

namespace internal {

// These are defined in:
// https://developers.google.com/protocol-buffers/docs/encoding
enum WireType {
    WT_VARINT = 0,            // int32,int64,uint32,uint64,sint32,sin64,bool,enum
    WT_64BIT = 1,             // fixed64,sfixed64,double
    WT_LENGTH_DELIMITED = 2,  // string,bytes,embedded messages,packed repeated fields
    WT_GROUP_START = 3,       // Groups(deprecated)
    WT_GROUP_END = 4,         // Groups(deprecated)
    WT_32BIT = 5,             // fixed32,sfixed32,float
};

template <typename From, typename To>
class is_convertible {
    typedef char one;
    typedef int two;

    template <typename To1>
    static To1& create();

    template <typename To1>
    static one test(To1);

    template <typename>
    static two test(...);

 public:
    static const bool value = (sizeof(test<To>(create<From>())) == sizeof(one));
};

template <class T>
struct is_integral {
    static const bool value = false;
};
template <>
struct is_integral<bool> {
    static const bool value = true;
};
template <>
struct is_integral<int32_t> {
    static const bool value = true;
};
template <>
struct is_integral<uint32_t> {
    static const bool value = true;
};
template <>
struct is_integral<int64_t> {
    static const bool value = true;
};
template <>
struct is_integral<uint64_t> {
    static const bool value = true;
};
template <>
struct is_integral<uint16_t> {
    static const bool value = true;
};
template <>
struct is_integral<int16_t> {
    static const bool value = true;
};
template <>
struct is_integral<uint8_t> {
    static const bool value = true;
};
template <>
struct is_integral<int8_t> {
    static const bool value = true;
};
template <>
struct is_integral<float> {
    static const bool value = true;
};
template <>
struct is_integral<double> {
    static const bool value = true;
};

template <class T>
struct is_enum {
    static const bool value = is_convertible<T, int32_t>::value & !is_integral<T>::value;
};

template <typename T, bool isEnum = is_enum<T>::value>
struct isMessage {
    enum { WRITE_TYPE = WT_LENGTH_DELIMITED };
};
template <typename T>
struct isMessage<T, true> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<std::string> {
    enum { WRITE_TYPE = WT_LENGTH_DELIMITED };
};
template <>
struct isMessage<bool> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<int32_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<uint32_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<uint16_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<int16_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<uint8_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<int8_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<int64_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<uint64_t> {
    enum { WRITE_TYPE = WT_VARINT };
};
template <>
struct isMessage<float> {
    enum { WRITE_TYPE = WT_32BIT };
};
template <>
struct isMessage<double> {
    enum { WRITE_TYPE = WT_64BIT };
};

template <class T, class C>
FORCEINLINE void serialize(T& t, C& c) {
    c.serialize(t);
}

template <class T, class C>
FORCEINLINE void serializeWrapper(T& t, C& c) {
    serialize(t, c);
}

template <typename T, bool isEnum = is_enum<T>::value>
struct TypeTraits {
    typedef T Type;
};
template <typename T, bool isEnum>
struct TypeTraits<std::vector<T>, isEnum> {
    typedef std::vector<T> Type;
};
template <typename T>
struct TypeTraits<T, true> {
    typedef int32_t Type;
};

namespace STOT {
template <typename T>
struct type {};
const int32_t _RTBUFSIZE = 128;
static char szTransbuf[_RTBUFSIZE] = {0};
// bool
template <>
struct type<bool> {
    static inline bool strto(const char* str) { return atoi(str) != 0; }
    static inline const char* tostr(bool v) { return v ? "1" : "0"; }
};
// int32_t
template <>
struct type<int32_t> {
    static inline int32_t strto(const char* str) { return (int32_t)atoi(str); }
    static inline const char* tostr(int32_t v) {
        snprintf(szTransbuf, sizeof(int32_t), "%d", v);
        return szTransbuf;
    }
};
// uint32_t
template <>
struct type<uint32_t> {
    static inline uint32_t strto(const char* str) { return (uint32_t)atoi(str); }
    static inline const char* tostr(uint32_t v) {
        snprintf(szTransbuf, sizeof(uint32_t), "%u", v);
        return szTransbuf;
    }
};

// int16_t
template <>
struct type<int16_t> {
    static inline int16_t strto(const char* str) { return (int16_t)atoi(str); }
    static inline const char* tostr(int16_t v) {
        snprintf(szTransbuf, sizeof(int16_t), "%d", v);
        return szTransbuf;
    }
};
// uint16_t
template <>
struct type<uint16_t> {
    static inline uint16_t strto(const char* str) { return (uint16_t)atoi(str); }
    static inline const char* tostr(uint16_t v) {
        snprintf(szTransbuf, sizeof(uint16_t), "%u", v);
        return szTransbuf;
    }
};

// int8_t
template <>
struct type<int8_t> {
    static inline int8_t strto(const char* str) { return (int8_t)atoi(str); }
    static inline const char* tostr(int8_t v) {
        snprintf(szTransbuf, sizeof(int8_t), "%d", v);
        return szTransbuf;
    }
};
// uint8_t
template <>
struct type<uint8_t> {
    static inline uint8_t strto(const char* str) { return (uint8_t)atoi(str); }
    static inline const char* tostr(uint8_t v) {
        snprintf(szTransbuf, sizeof(uint8_t), "%u", v);
        return szTransbuf;
    }
};

// int64_t
template <>
struct type<int64_t> {
    static inline int64_t strto(const char* str) { return atoll(str); }
    static inline const char* tostr(int64_t v) {
        snprintf(szTransbuf, sizeof(int64_t), "%lld", (long long)v);  // NOLINT
        return szTransbuf;
    }
};
// uint64_t
template <>
struct type<uint64_t> {
    static inline uint64_t strto(const char* str) { return (uint64_t)atoll(str); }
    static inline const char* tostr(uint64_t v) {
        snprintf(szTransbuf, sizeof(uint64_t), "%llu", (long long)v);  // NOLINT
        return szTransbuf;
    }
};
// std::string
template <>
struct type<std::string> {
    static inline const std::string& strto(const std::string& str) { return str; }
    static inline const char* tostr(const std::string& v) { return v.c_str(); }
};

}  // namespace STOT

}  // namespace internal

}  // namespace struct2x

#endif  // MIDDLEWARE_PER_STRUCT2X_INCLUDE_STRUCT2X_TRAITS_H_
