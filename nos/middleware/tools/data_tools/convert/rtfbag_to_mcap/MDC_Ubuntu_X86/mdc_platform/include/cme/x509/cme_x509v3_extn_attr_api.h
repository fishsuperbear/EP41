/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/10/26
 * Notes:
 * History:
 * 2020/10/26 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_x509v3_extn_attr CME_X509V3_EXTN_ATTR_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_ATTR_API_H
#define CME_X509V3_EXTN_ATTR_API_H

#include "cme_asn1_api.h"
#include "cme_cid_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief OtherName结构是GeneralName的一部分。为最终实体证书 Subject 的标记唯一标识符的其他名称。
 *
 */
typedef struct {
    Asn1Oid id;            /* < ASN1 OID */
    Asn1OctetString *type; /* < ASN1不支持的类型用OctetString保存数据，不参与编解码 */
} X509ExtOthName;

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief X509ExtNameStrType 枚举定义了支持的不同字符串格式
 *
 */
typedef enum {
    X509EXT_STR_FORMAT_TELETEX = 0, /* < TELETEX 字符串 */
    X509EXT_STR_FORMAT_PRTABLE = 1, /* < PRINTABLE 字符串 */
    X509EXT_STR_FORMAT_UNI = 2,     /* < UNIVERSAL 字符串 */
    X509EXT_STR_FORMAT_UTF8 = 3,    /* < UTF8 字符串 */
    X509EXT_STR_FORMAT_BMP = 4,     /* < BMP 字符串 */
    X509EXT_STR_FORMAT_VISIBLE = 5, /* < VISIBLE 字符串 */
    X509EXT_STR_FORMAT_NUM = 6,     /* < NUMERIC 字符串 */
    X509EXT_STR_FORMAT_IA5 = 7      /* < IA5 字符串 */
} X509ExtNameStrType;

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief X500 Name中的属性类型
 *
 */
typedef enum {
    X509EXT_TYPE_COUNTRY = CID_AT_COUNTRYNAME,                  /* < Country Name */
    X509EXT_TYPE_ORG = CID_AT_ORGANIZATIONNAME,                 /* < Organization Name */
    X509EXT_TYPE_ORG_UNIT_NAME = CID_AT_ORGANIZATIONALUNITNAME, /* < Organization Unit Name */
    X509EXT_TYPE_DNQUALIFIER = CID_AT_DNQUALIFIER,              /* < DN Qualifier */
    X509EXT_TYPE_STATE = CID_AT_STATEORPROVINCENAME,            /* < State or Province name */
    X509EXT_TYPE_NAME = CID_AT_NAME,                            /* < Name */
    X509EXT_TYPE_COMMON_NAME = CID_AT_COMMONNAME,               /* < Common Name */
    X509EXT_TYPE_SERIAL_NUMBER = CID_AT_SERIALNUMBER,           /* < Serial Number */
    X509EXT_TYPE_LOCALITY = CID_AT_LOCALITYNAME,                /* < Locality name */
    X509EXT_TYPE_TITLE = CID_AT_TITLE,                          /* < Title */
    X509EXT_TYPE_SURNAME = CID_AT_SURNAME,                      /* < Surname */
    X509EXT_TYPE_GIVENNAME = CID_AT_GIVENNAME,                  /* < Given Name */
    X509EXT_TYPE_INITIALS = CID_AT_INITIALS,                    /* < Initial */
    X509EXT_TYPE_PSEUDONYM = CID_AT_PSEUDONYM,                  /* < Psuedonym */
    X509EXT_TYPE_GENQUALIFIER = CID_AT_GENERATIONQUALIFIER,     /* < Generation Qualifier */
    X509EXT_TYPE_EMAIL = CID_EMAILADDRESS,                      /* < Email Address */
    X509EXT_TYPE_USER_ID = CID_USER_ID,                         /* < user id */
    X509EXT_TYPE_HOST = CID_HOST,                               /* < host */
    X509EXT_TYPE_DOMAIN_COMPONENT = CID_DOMAINCOMPONENT,        /* < Domain Component */
    X509EXT_TYPE_DNS_NAME = CID_DNS_NAME,                       /* < DNS name */
    X509EXT_TYPE_POSTAL_CODE = CID_POSTAL_CODE,                 /* < postal code */
    X509EXT_TYPE_STREET_ADDRESS = CID_AT_STREETADDRESS,         /* < Street Address */
    X509EXT_TYPE_UNSTRUCTURED_NAME = CID_UNSTRUCTURED_NAME,     /* < UnstructuredName */
    X509EXT_TYPE_UNSTRUCTURED_ADDR = CID_UNSTRUCTURED_ADDR      /* < UnstructuredAddress */
} X509ExtNameAttrType;

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief X509Extension Attribute Type Value
 *
 */
typedef struct {
    Asn1Oid type;            /* < 属性的对象标识符 */
    Asn1AnyDefinedBy anyVal; /* < 此字段包含实际标识信息 */
} X509ExtAttrTypeValue;

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief X509 Extension Attribute.
 * @attention 此结构用作保存相同类型值的证书扩展的一部分。
 *
 */
typedef struct {
    Asn1Oid type;      /* < 需要设置的Attribute类型 */
    ListHandle values; /* < 包含相同类型值的列表 */
} X509ExtAttr;

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief X509 Extension Attribute List.
 *
 */
typedef CmeList X509ExtAttributeList;

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief subject属性扩展。
 *
 */
typedef enum {
    X509EXT_TYPE_DOB = CID_PDA_DATEOFBIRTH,                           /* < 出生时间 */
    X509EXT_TYPE_PLACEOFBIRTH = CID_PDA_PLACEOFBIRTH,                 /* < 出生地点 */
    X509EXT_TYPE_GENDER = CID_PDA_GENDER,                             /* < 性别 */
    X509EXT_TYPE_COUNTRYOFCITIZENSHIP = CID_PDA_COUNTRYOFCITIZENSHIP, /* < 国籍 */
    X509EXT_TYPE_COUNTRYOFRESIDENCE = CID_PDA_COUNTRYOFRESIDENCE      /* < 居住地 */
} X509ExtSubDirAttr;

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief 创建属性type & value结构。
 * @par 描述：
 * 此函数创建在Name结构中的X509ExtAttrTypeValue结构。\n
 * AttributeTypeAndValue结构包含一个OID和AnyDefinedBy类型，
 * 其中必须复制数据。根据OID确定AnyDefinedBy类型中的值。
 *
 * @param attrID [IN] attrID 必须为其创建结构的CID。
 * @param strFormat [IN] stringFormat 存储结构中的数据的字符串格式。
 * 字符串格式多为Asn1PrintableString。Email、DomainComponent在IA5String中表示。
 * X509ExtNameStrType中包含所有要使用的字符串格式。
 * @param attrValueLen [IN] 要放入AttributeTypeAndValue结构成员中的输入数据的长度。
 * @param attrValue [IN] 必须在AttributeTypeAndValue结构的AnyDefinedBy字段中设置的字符串。
 * 字符串类型除可打印字符串外，不验证字符串类型。
 *
 * @retval X509ExtAttrTypeValue 成功执行时返回X509ExtAttrTypeValue结构的指针。
 * @retval NULL 失败返回NULL，故障条件可以是以下原因：
 * @li 输入参数为NULL。
 * @li 内存申请失败。
 * @li 属性类型不存在。
 */
X509ExtAttrTypeValue *X509EXT_AttrTypeAndValueCreate(X509ExtNameAttrType attrID, X509ExtNameStrType strFormat,
                                                     size_t attrValueLen, const uint8_t *attrValue);

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief 深拷贝X509ExtAttrTypeValue结构体。
 *
 * @par 描述：
 * 深拷贝X509ExtAttrTypeValue结构体。
 * @param [IN] 指向X509ExtAttrTypeValue结构的源指针。
 *
 * @retval X509ExtAttrTypeValue* 指向X509ExtAttrTypeValue结构的目的指针。
 * @retval NULL 输入参数为NULL。
 * @retval NULL 内存申请失败。
 */
X509ExtAttrTypeValue *X509EXT_AttrTypeAndValueDump(const X509ExtAttrTypeValue *src);

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief 释放X509ExtAttrTypeValue结构体。
 *
 * @param attr [IN] 指向待释放的X509ExtAttrTypeValue结构指针。
 *
 * @retval void 不返回任何值。
 */
void X509EXT_AttrTypeAndValueFree(X509ExtAttrTypeValue *attr);

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief 深拷贝X509ExtAttr结构。
 *
 * @param src [IN] 指向X509ExtAttr的源指针。
 * @retval X509ExtAttr* 指向X509ExtAttr的目的指针。
 */
X509ExtAttr *X509EXT_AttributeDump(const X509ExtAttr *src);

/**
 * @ingroup cme_x509v3_extn_attr
 * @brief 释放X509ExtAttr结构体。
 *
 * @param attr [IN] 待释放的X509ExtAttr结构体指针。
 */
void X509EXT_AttributeFree(X509ExtAttr *attr);

#ifdef __cplusplus
}
#endif

#endif // CME_X509V3_EXTN_ATTR_API_H
