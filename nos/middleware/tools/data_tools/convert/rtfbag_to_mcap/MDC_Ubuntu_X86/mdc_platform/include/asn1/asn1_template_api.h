/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: ASN1功能对组件的类型定义
 * Create: 2020/10/15
 * Notes:
 * History:
 * 2020/10/15 第一次创建
 */

#ifndef ASN1_TEMPLATE_API_H
#define ASN1_TEMPLATE_API_H

#include <inttypes.h>
#include <stddef.h>
#include <unistd.h>
#include <limits.h>
#include "asn1_types_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup asn1_type
 * @brief Asn1ContextHandle ASN1编解码上下文管理句柄
 */
typedef struct Asn1Context_ *Asn1ContextHandle;
typedef const struct Asn1Context_ *Asn1ContextRoHandle;

#define ASN1_FLAG_NORMAL 0u        /* < 表示用户管理内存 */
#define ASN1_FLAG_POINTER 1u       /* < 表示模块管理内存 */
#define ASN1_FLAG_DEFAULT 2u       /* < 表示节点可使用默认值 */
#define ASN1_FLAG_OPTIONAL 4u      /* < 表示节点可选（可不存在该节点） */
#define ASN1_FLAG_ANY_OID 8u       /* < 表示该节点是任意节点，索引是OID，可替换为其他节点 */
#define ASN1_FLAG_ANY_IDX 16u      /* < 表示该节点是任意节点,索引时int型，可替换为其他节点 */
#define ASN1_FLAG_RANGE 32u        /* < 表示该节点指定了范围 */
#define ASN1_FLAG_EXPLICIT 64u     /* < 表示该节点是显示标签 */
#define ASN1_FLAG_IMPLICIT 128u    /* < 表示该节点是隐式标签 */
#define ASN1_FLAG_APPLICATION 256u /* < 表示该节点是应用自定义节点 */

#define ASN1_FLAG_NO_TAG (ASN1_FLAG_EXPLICIT | ASN1_FLAG_IMPLICIT) /* < 表示该节点是显示或者隐式标签 */
#define ASN1_FLAG_ANY (ASN1_FLAG_ANY_OID | ASN1_FLAG_ANY_IDX)      /* < 表示该节点是任意类型 */
#define ASN1_FLAG_IGNORE (ASN1_FLAG_DEFAULT | ASN1_FLAG_OPTIONAL)  /* < 表示该节点是默认或者可选的 */

/**
 * @ingroup asn1_type
 * @brief Asn1TagClass ASN1节点标签值
 */
#define ASN1_CLASS_UNIVERSAL 0u /* < 该值的含义在所有的应用中都相同。这种类型只在X.208中定义 */
#define ASN1_CLASS_APPLICATION 1u /* < 该值的含义由应用决定，如X.500目录服务 */
#define ASN1_CLASS_CONTEXT 2u     /* < 该值的含义根据给定的结构类型而不同 */
#define ASN1_CLASS_PRIVATE 3u     /* < 该值的含义根据给定的企业而不同 */

/**
 * @ingroup asn1_type
 * @brief ASN1节点结构类型
 */
#define ASN1_ENCODING_PRIMITIVE 0u   /* < 表示节点为简单类型 */
#define ASN1_ENCODING_CONSTRUCTED 1u /* < 表示节点为结构类型 */

/**
 * @ingroup asn1_type
 * @brief节点的类型值
 */
#define ASN1_TAG_EOC 0u              /* < 无效类型 */
#define ASN1_TAG_BOOL 1u             /* < 布尔类型 */
#define ASN1_TAG_INT 2u              /* < 整型数值 */
#define ASN1_TAG_BITSTRING 3u        /* < 比特流串 */
#define ASN1_TAG_OCTETSTRING 4u      /* < 字节流串 */
#define ASN1_TAG_NULL 5u             /* < 空类型 */
#define ASN1_TAG_OID 6u              /* < OID（对象标识符） */
#define ASN1_TAG_ENUMERATED 10u      /* < 枚举类型 */
#define ASN1_TAG_UTF8STRING 12u      /* < UTF8字符串 */
#define ASN1_TAG_SEQUENCE 16u        /* < 序列 */
#define ASN1_TAG_SET 17u             /* < 集合 */
#define ASN1_TAG_NUMERICSTRING 18u   /* < 数字字符串 */
#define ASN1_TAG_PRINTABLESTRING 19u /* < 可打印字符串 */
#define ASN1_TAG_T61STRING 20u       /* < T16字符串 */
#define ASN1_TAG_IA5STRING 22u       /* < IA5字符串 */
#define ASN1_TAG_UTCTIME 23u         /* < UTC时间 */
#define ASN1_TAG_GENERALIZEDTIME 24u /* < 通用时间(ISO8601) */
#define ASN1_TAG_VISIBLESTRING 26u   /* < 可显示字符串 */
#define ASN1_TAG_UNIVERSALSTRING 28u /* < 通用字符串 */
#define ASN1_TAG_BMPSTRING 30u       /* < 双字节字符串(UNICODE STRING) */

/**
 * @ingroup asn1_type
 * @brief 生成ASN1节点类型字段
 *
 * @param class [IN] ASN1节点标签值.
 * @param encoding [IN] ASN1节点结构类型.
 * @param tag [IN] ASN1节点类型值.
 *
 * @return 节点类型字段
 */
#define ASN1_MAKE_TYPE(class, encoding, tag) \
    ((uint8_t)((uint8_t)(class) << 6u) | (uint8_t)((encoding) << 5u) | (uint8_t)(tag))

typedef struct Asn1Item_ Asn1Item;

/**
 * @ingroup asn1_type
 * @brief 节点指针函数，指向已定义的ASN1节点
 *
 * @return 节点地址
 */
typedef Asn1Item *(*Asn1GetItemFunc)(void);

/**
 * @ingroup asn1_type
 * @brief ASN1数据的限制范围
 */
typedef struct {
    Asn1Int min;   /* < 范围的最小值 */
    Asn1Int max;   /* < 范围的最大值 */
    Asn1Int extId; /* < 扩展ID */
} Asn1Range;

/**
 * @ingroup asn1_type
 * @brief ASN1模板
 */
typedef struct {
    Asn1GetItemFunc getItem; /* < ASN1节点 */
    uint8_t idx;             /* < ASN1选择索引(当tagType为隐式或显示标签时有效) */
    uint16_t flag;           /* < ASN1节点标记 */
    size_t dataOffset;       /* < ASN1节点字段的偏移量（字段在父结构体中的偏移量） */
    size_t asnAnyOffset;     /* < ASN1节点Any索引的偏移量（Any索引字段在父结构体中的偏移量） */
    void *defaultValue;      /* < ASN1节点的默认值 */
    Asn1Range range;         /* < ASN1节点的范围 */
    const char *traceInfo;   /* < ASN1节点的跟踪信息 */
} Asn1Template;

/**
 * @ingroup asn1_type
 * @brief 模板指针函数，指向已定义的ASN1模板
 *
 * @return 模板地址
 */
typedef Asn1Template *(*Asn1GetTemplateFunc)(void);

typedef enum {
    ASN1_FUNC_BOOLEAN,
    ASN1_FUNC_ENUMERATED,
    ASN1_FUNC_INTEGER,
    ASN1_FUNC_BIGINT,
    ASN1_FUNC_SIGNEDBIGINT,
    ASN1_FUNC_NULL,
    ASN1_FUNC_BITSTRING,
    ASN1_FUNC_OCTETSTRING,
    ASN1_FUNC_FIXEDSTRING,
    ASN1_FUNC_SEQ,
    ASN1_FUNC_SEQOF,
    ASN1_FUNC_SET,
    ASN1_FUNC_SETOF,
    ASN1_FUNC_CHOICE,
    ASN1_FUNC_UNKNOWN,
    ASN1_FUNC_ANY,
    ASN1_FUNC_PLACEHOLDER,
    ASN1_FUNC_FORWARD,
    ASN1_FUNC_BUFF,
} Asn1FuncType;

/**
 * @ingroup asn1_type
 * @brief ASN1节点
 */
struct Asn1Item_ {
    Asn1GetTemplateFunc getTemplate; /* < ASN1模板数组 */
    Asn1FuncType funcType;           /* < ASN1编解码函数 */
    uint8_t tag;                     /* < ASN1节点值 */
    uint8_t tempCount;               /* < ASN1模板数组的大小 */
    size_t dataSize;                 /* < ASN1节点绑定的结构体大小 */
};

Asn1Item *ASN1_ItemInt(void);

Asn1Item *ASN1_ItemEnumerated(void);

Asn1Item *ASN1_ItemBigInt(void);

Asn1Item *ASN1_ItemSignedBigInt(void);

Asn1Item *ASN1_ItemBool(void);

Asn1Item *ASN1_ItemNull(void);

Asn1Item *ASN1_ItemBitString(void);

Asn1Item *ASN1_ItemOID(void);

Asn1Item *ASN1_ItemNumericString(void);

Asn1Item *ASN1_ItemOctetString(void);

Asn1Item *ASN1_ItemUTF8String(void);

Asn1Item *ASN1_ItemPrintable(void);

Asn1Item *ASN1_ItemT16String(void);

Asn1Item *ASN1_ItemIA5String(void);

Asn1Item *ASN1_ItemVisible(void);

Asn1Item *ASN1_ItemUniversalString(void);

Asn1Item *ASN1_ItemBMPString(void);

Asn1Item *ASN1_ItemUTCTime(void);

Asn1Item *ASN1_ItemGeneralizedTime(void);

Asn1Item *ASN1_ItemPlaceholder(void);

Asn1Item *ASN1_ItemUnknown(void);

#ifdef __cplusplus
}
#endif /* extern 'C' */

#endif // ASN1_TEMPLATE_API_H
