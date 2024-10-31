/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: ASN1功能的对外类型定义
 * Create: 2020/10/15
 * Notes:
 * History:
 * 2020/10/15 第一次创建
 */
#ifndef ASN1_TYPES_API_H
#define ASN1_TYPES_API_H

#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include "cstl_list.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup asn1 asn1 */
/** @defgroup asn1_type ASN1模块类型定义
 * @ingroup asn1
 */
/**
 * @ingroup asn1_type
 * pkey 最大大数长度为 1024
 */
#define PKEY_BIGINT_MAX_LEN 1024u /* < pkey 最大大数长度为 1024 */

/**
 * @ingroup asn1_type
 * Asn1BigInt 结构体， 用于传递 PKey
 */
typedef struct {
    size_t len;                       /* < bigInt 有效字节数 */
    uint8_t val[PKEY_BIGINT_MAX_LEN]; /* < MSB 方式保存 bigInt */
} Asn1BigInt;

/**
 * @ingroup asn1_type
 * @brief 字节串
 */
typedef struct {
    size_t len;    /* < buffer长度 */
    uint8_t *data; /* < buffer数据 */
} Asn1OctetString;

/**
 * @ingroup asn1_type
 * @brief 比特串
 */
typedef struct {
    size_t bitLength;  /* < 比特串长度 */
    uint8_t *data;     /* < 比特串数据 */
    size_t byteLength; /* < 比特字节长度 */
} Asn1BitString;

/**
 * @ingroup asn1_type
 * Asn1List ASN1任意类型
 */
typedef CstlList *ListHandle;
typedef const CstlList *ListRoHandle;

/**
 * @ingroup asn1_type
 * Asn1Any ASN1任意类型
 */
typedef void *Asn1Any;
/**
 * @ingroup asn1_type
 * Asn1AnyDefinedBy 依靠其他字段确定类型的ASN1任意类型
 */
typedef Asn1Any Asn1AnyDefinedBy;
/**
 * @ingroup asn1_type
 * Asn1Int ASN1整型数据
 */
typedef ssize_t Asn1Int;

/**
 * @ingroup asn1_type
 * Asn1Null ASN1空类型
 */
typedef char Asn1Null;
/**
 * @ingroup asn1_type
 * Asn1Oid ASN1对象描述符
 */
typedef Asn1OctetString Asn1Oid;
/**
 * @ingroup asn1_type
 * Asn1VisibleString ASN1可视字符串
 */
typedef Asn1OctetString Asn1VisibleString;
/**
 * @ingroup asn1_type
 * Asn1VisibleString ASN1 Teletex字符串
 */
typedef Asn1OctetString Asn1TeletexString;
/**
 * @ingroup asn1_type
 * Asn1VisibleString ASN1 UTF8字符串
 */
typedef Asn1OctetString Asn1UTF8String;
/**
 * @ingroup asn1_type
 * Asn1VisibleString ASN1通用字符串
 */
typedef Asn1OctetString Asn1UniversalString;
/**
 * @ingroup asn1_type
 * Asn1VisibleString ASN1可打印字符串
 */
typedef Asn1OctetString Asn1PrintableString;
/**
 * @ingroup asn1_type
 * Asn1VisibleString ASN1数字字符串
 */
typedef Asn1OctetString Asn1NumericString;

/**
 * @ingroup asn1_type
 * Asn1VisibleString ASN1 IA5字符串
 */
typedef Asn1OctetString Asn1IA5String;
/**
 * @ingroup asn1_type
 * Asn1BMPString ASN1 BMP字符串
 */
typedef Asn1OctetString Asn1BMPString;
/**
 * @ingroup asn1_type
 * Asn1UTCTime ASN1世界标准时间
 */
typedef Asn1OctetString Asn1UTCTime;
/**
 * @ingroup asn1_type
 * Asn1GeneralizedTime ASN1通用时间
 */
typedef Asn1OctetString Asn1GeneralizedTime;

#ifdef __cplusplus
}
#endif

#endif // ASN1_TYPES_API_H
