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
/** @defgroup cme_x509v3_extn_dn CME_X509V3_EXTN_DN_API
 * @ingroup cme_x509
 */
#ifndef CME_X509V3_EXTN_DN_API_H
#define CME_X509V3_EXTN_DN_API_H

#include "stdbool.h"
#include "cme_x509v3_extn_attr_api.h"
#include "cme_asn1_api.h"
#include "crypto/crypto_hash_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief RelativeDN 序列。
 *
 */
typedef CmeList X509ExtRDNSeq;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief 通用名称结构列表。
 *
 */
typedef CmeList X509ExtGenNameList; /* < X509ExtGenName */

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief X509 Extension DN Attribute Value.
 *
 */
typedef struct {
    X509ExtNameAttrType attrType; /* < attrType 枚举 */
    uint8_t *attrValue;           /* < 与CID对应的字符串 */
    X509ExtNameStrType strType;   /* < 表示属性值的字符串类型。 */
} X509ExtDnAttrValue;

typedef CmeList X509ExtDnAttrValueList; /* < Asn1OctetString */

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief X509 Extension Name ID.
 *
 */
typedef enum {
    NAME_RDNSEQ /* < 选择指示RDNSequence */
} X509ExtNameId;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief X509ExtName 结构体，用于标识证书颁发者和证书使用者的所有证书。名称是RDNSequence的列表
 */
typedef struct {
    X509ExtNameId choiceId; /* < 用于指示RDNSequence的选项 */
    union NameChoice {
        X509ExtRDNSeq *rdnSeq; /* < 包含证书使用者或颁发者信息的列表 */
    } a;                       /* < 包含RDNSequence列表的联合 */
} X509ExtName;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   创建X509名称结构。
 * @param   dnAttrLen [IN] 结构体长度。
 * @param   dnAttrValue [IN] 包含属性、stringtype和数据的CID的结构体。
 * @retval  X509ExtName，指向创建的名称的指针。
 */
X509ExtName *X509EXT_NameCreate(size_t dnAttrLen, const X509ExtDnAttrValue *dnAttrValue);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   比较两个 X509ExtName 结构，两个名称完全相等时，才返回相等。
 * @param   firstName [IN] 第一个 X509ExtName 结构。
 * @param   secondName [IN] 第二个 X509ExtName 结构。
 * @retval  true 名称相同
 * @retval  false 名称不同
 */
bool X509EXT_NameCompare(const X509ExtName *firstName, const X509ExtName *secondName);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   对给定的 X509ExtName 结构进行编码。
 * @param   name [IN] 要编码的名称。
 * @param   encodedLen [OUT] 编码后数据的长度。
 * @retval  uint8_t *，对名称编码后的指针地址。
 */
uint8_t *X509EXT_NameEncode(const X509ExtName *name, size_t *encodedLen);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   复制一份 X509ExtName 结构。
 * @param   srcName [IN] 待复制的X509ExtnName。
 * @retval  已复制的X509ExtnName指针。
 */
X509ExtName *X509EXT_NameDump(const X509ExtName *srcName);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   从编码的名称缓存区生成解码后的 X509ExtName 结构。
 * @param   derCodeOfName [IN] 要解码的名称缓存区。
 * @param   encNameLen [IN] 缓存区长度。
 * @param   decodedLen [OUT] 解码后数据的长度。
 * @retval  X509ExtName，解码后的名称结构体指针。
 */
X509ExtName *X509EXT_NameDecode(const uint8_t *derCodeOfName, size_t encNameLen, size_t *decodedLen);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief 比较两个 X509ExtName结构。
 * 若第二个名称参数全部包含第一个名称的所有名称，则判断为相等。（完全比较，忽略顺序）。
 * @param   firstName [IN] 第一个 X509ExtName 结构。
 * @param   secondName [IN] 第二个 X509ExtName 结构。
 * @retval  返回 X509ExtName 比较结果，相等返回CME_SUCCESS。
 */
bool X509EXT_NameCompleteCompare(const X509ExtName *firstName, const X509ExtName *secondName);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   使用给定的哈希算法计算名称结构的哈希。
 * @par 描述：
 * 生成名称的哈希结构，对名称进行编码，然后根据输入的哈希算法计算其中的哈希值。哈希值可用于验证名字。
 * @param   name [IN] 名称结构体。
 * @param   hashAlg [IN] 哈希算法。
 * @param   hashLen [OUT] 哈希值长度。
 * @retval  uint8_t *，计算哈希后的哈希值缓存指针。
 */
uint8_t *X509EXT_NameCalcHash(const X509ExtName *name, CryptoHashAlgorithm hashAlg, size_t *hashLen);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   提取给定名称结构中特定属性的值,属性类型可以是名称结构中支持的任何类型。
 * @param   attrType [IN] 名称结构中的属性类型。
 * @param   name [IN] 待检索信息的名称。
 * @param   strType [OUT] 用于在名称结构中表示的字符串类型。
 * @retval  返回指向名称结构中给定属性类型的属性值的指针。
 */
Asn1OctetString *X509EXT_NameGetAttr(X509ExtNameAttrType attrType, const X509ExtName *name,
                                     X509ExtNameStrType *strType);
/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   为DN设置新的attribute。
 * @param   name [IN] DN.
 * @param   dnAttrValue [IN] 包含属性、stringtype和数据的CID的结构体。
 * @param   dnAttrLen [IN] 结构体长度。
 * @retval error code.
 */
int32_t X509EXT_NameSetAttr(X509ExtName *name, const X509ExtDnAttrValue *dnAttrValue, size_t dnAttrLen);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   删除给定名称结构中特定属性的值,属性类型可以是名称结构中支持的任何类型。
 * @param   attrType [IN] 名称结构中的属性类型。
 * @param   name [IN] 待检索信息的名称。
 * @retval error code.
 */
int32_t X509EXT_NameRemoveAttr(X509ExtNameAttrType attrType, const X509ExtName *name);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   提取给定名称结构中特定属性的值列表,属性类型可以是名称结构中支持的任何类型。
 * @param   attrType [IN] 名称结构中的属性类型。
 * @param   name [IN] 待检索信息的名称。
 * @retval  返回指向名称结构中给定属性类型的属性值的列表。
 */
X509ExtDnAttrValueList *X509EXT_NameGetAttrList(X509ExtNameAttrType attrType, const X509ExtName *name);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   释放 X509ExtName。
 * @param   name [IN] 待释放的X509Extn_Name。
 */
void X509EXT_NameFree(X509ExtName *name);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief 枚举用于提供在证书策略中创建策略限定符的选项。
 * @par 描述：
 * 此枚举定义用于创建ORADDRESS结构、EDI参与方名称、URL、RFC822名称、DNS名称、IP地址、
 * Regester id、其他名称的所有常量，这些常量将在内部用于创建通用名称。
 * 所有这些常量都需要用于创建公共节点，这些节点内部将形成一个公共节点列表，
 * 并作为输入提供，以创建general name Api。
 *
 */
typedef enum {
    X509EXT_TYPE_ORADDR = 0,                        /* < 创建OrAddress结构 */
    X509EXT_TYPE_BUILTINSTDATTR = 1,                /* < 创建内置标准属性结构 */
    X509EXT_TYPE_COUNTRYNAME = 2,                   /* < 设置国家名称 */
    X509EXT_TYPE_ADMINDOMAIN = 3,                   /* < 设置管理域名 */
    X509EXT_TYPE_NETADDR = 4,                       /* < 设置网络地址 */
    X509EXT_TYPE_TERMINALID = 5,                    /* < 设置终端标识符 */
    X509EXT_TYPE_PRIDOMAIN = 6,                     /* < 设置专用域名 */
    X509EXT_TYPE_ORGNAME = 7,                       /* < 设置组织名称 */
    X509EXT_TYPE_NUMID = 8,                         /* < 设置数字用户标识符 */
    X509EXT_TYPE_PERSONALNAME = 9,                  /* < 设置个人姓名 */
    X509EXT_TYPE_ORGUNITNAME = 10,                  /* < 设置组织名称 */
    X509EXT_TYPE_BUILTINDOMAINDEFATTR = 11,         /* < 设置域定义的属性 */
    X509EXT_TYPE_EXTATTR = 12,                      /* < 设置扩展属性 */
    X509EXT_TYPE_EDIPARTYNAME = 13,                 /* < 创建 EDI Party名称 */
    X509EXT_TYPE_NAMEASSIGNER = 14,                 /* < 设置转让人名称 */
    X509EXT_TYPE_PARTYNAME = 15,                    /* < 创建 Party名称 */
    X509EXT_TYPE_RFC822NAME = 16,                   /* < 设置RFC822名称 */
    X509EXT_TYPE_DNSNAME = 17,                      /* < 设置DNS名称 */
    X509EXT_TYPE_IPADDRESS = 18,                    /* < 设置IP地址 */
    X509EXT_TYPE_REGID = 19,                        /* < 设置 RegisteredID */
    X509EXT_TYPE_URL = 20,                          /* < 设置URL */
    X509EXT_TYPE_OTHNAME = 21,                      /* < 创建 other 名称 */
    X509EXT_TYPE_IDVALUE = 22,                      /* < 设置 other ID 值 */
    X509EXT_TYPE_ASSIGNER = 23,                     /* < 设置 other 转让人 */
    X509EXT_TYPE_STR = 24,                          /* < 设置 other 字符串 */
    X509EXT_TYPE_ORADDR_SURNAME = 25,               /* < 设置 surName */
    X509EXT_TYPE_ORADDR_GENQUALIFIER = 26,          /* < 设置 genqualifier */
    X509EXT_TYPE_ORADDR_INITIALS = 27,              /* < 设置 initials */
    X509EXT_TYPE_ORADDR_GIVENNAME = 28,             /* < 设置 giveName */
    X509EXT_COMMON_NAME_ANY_ID = 29,                /* < 用于创建扩展属性的 CommonName 结构 */
    X509EXT_TELETEX_COMMON_NAME_ANY_ID = 30,        /* < 用于创建扩展属性的 Teletex CommonName 结构 */
    X509EXT_TELETEX_ORG_NAME_ANY_ID = 31,           /* < 用于创建扩展属性的组织名称结构 */
    X509EXT_TELETEX_PERSONAL_NAME_ANY_ID = 32,      /* < 用于创建扩展属性的人名结构 */
    X509EXT_TELETEX_ORG_UNIT_NAMES_ANY_ID = 33,     /* < 创建扩展属性的组织单元名称列表 */
    X509EXT_PDS_NAME_ANY_ID = 34,                   /* < 创建PDS名称 */
    X509EXT_PHYL_DELY_COUNTRY_NAME_ANY_ID = 35,     /* < 创建实际传输国家/地区名称 */
    X509EXT_POSTAL_CODE_ANY_ID = 36,                /* < 创建邮政地址结构 */
    X509EXT_PHYL_DELY_OFFICE_NAME_ANY_ID = 37,      /* < 创建物理传输办公室名称 */
    X509EXT_PHY_DELY_OFFICE_NUMBER_ANY_ID = 38,     /* < 创建实际传输办公室编号 */
    X509EXT_EXT_OR_ADDR_CMPT_ANY_ID = 39,           /* < 创建扩展名或地址结构 */
    X509EXT_PHY_DELY_PERSONAL_NAME_ANY_ID = 40,     /* < 创建实际传输国家/地区名称 */
    X509EXT_PHY_DELY_ORG_NAME_ANY_ID = 41,          /* < 创建物理传输组织名称 */
    X509EXT_EXT_PHYL_DELY_ADDR_CMPT_ANY_ID = 42,    /* < 创建物理传递地址组件 */
    X509EXT_UNFORMATTED_POSTAL_ADDRESS_ANY_ID = 43, /* < 创建未格式化的邮政地址 */
    X509EXT_ST_ADDR_ANY_ID = 44,                    /* < 创建街道地址 */
    X509EXT_POST_OFFICE_BOX_ADDR_ANY_ID = 45,       /* < 创建邮政信箱地址 */
    X509EXT_POSTE_RESTANTE_ADDRESS_ANY_ID = 46,     /* < 创建重新启动后地址 */
    X509EXT_UNIQUE_POSTAL_NAME_ANY_ID = 47,         /* < 创建唯一的邮政名称 */
    X509EXT_LOCAL_POSTAL_ATTRS_ANY_ID = 48,         /* < 创建本地邮政属性 */
    X509EXT_EXT_NETADDR_ANY_ID = 49,                /* < 创建扩展网络地址 */
    X509EXT_TERMINAL_TYPE_ANY_ID = 50,              /* < 创建终端类型 */
    X509EXT_EXTNETADDR_E1634ADDR = 51,              /* < 如果使用e163-4-地址 */
    X509EXT_EXTNETADDR_PSAPADDR = 52,               /* < 如果使用psap地址 */
    X509EXT_TYPE_ORGUNITNAME_LIST = 53,             /* < 创建组织单位名称列表 */
    X509EXT_TYPE_POLICYID = 54,                     /* < 用于创建证书策略 */
    X509EXT_TYPE_USERNOTICE = 55,                   /* < 创建UserNotice限定符 */
    X509EXT_TYPE_EXPLICITTEXT = 56,                 /* < 在通知引用中设置显式文本 */
    X509EXT_TYPE_ORGTXT = 57,                       /* < 在通知引用中设置组织文本 */
    X509EXT_TYPE_NOTICENUMBERS = 58,                /* < 设置通知编号 */
    X509EXT_TYPE_CPS = 59                           /* < 创建CPS URI限定符 */
} X509ExtComType;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief X509ExtComData 结构体，用于创建公共数据，这些数据在创建许多扩展时内部使用。
 *
 */
typedef struct {
    X509ExtComType type;        /* < X509ExtComType 类型 */
    uint8_t *data;              /* < 指向数据字符串的指针 */
    ListHandle dataList;        /* < 指向公共节点列表的指针 */
    X509ExtNameStrType strType; /* < 字符串的类型 */
} X509ExtComData;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief X509 Extension ComDataList
 *
 */
typedef CmeList X509ExtComDataList; /* < X509ExtComData */

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   创建证书扩展时使用的公共数据结构。
 * @par 描述：
 * 数据可以是字符串，也可以是数据项列表。
 * 如果dataString和dataList都存在，则不会创建该结构，而是返回 NULL。
 * @param   infoType [IN] 创建结构的信息类型
 * @param   dataStr [IN] 必须根据前一个参数的类型使用的以NULL结尾的字符串
 * @param   dataList [IN] 用于创建扩展名的信息类型的数据列表
 * @param   strType [IN] 字符串类型仅在 enInfoType 具有字符串并且必须在其中表示字符串的字符串类型时使用
 * @retval  X509ExtComData *, 返回指向X509ExtnCommonData结构的指针。不成功返回NULL
 */
X509ExtComData *X509EXT_ComDataCreate(X509ExtComType infoType, const uint8_t *dataStr, ListHandle dataList,
                                      X509ExtNameStrType strType);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   释放 X509ExtComData 的公共数据结构。
 * @param   comdata [IN] 待释放的 X509ExtComData
 */
void X509EXT_ComDataFree(X509ExtComData *comdata);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief GeneralName提供了几个可以提供这些信息的选项。
 * @par 描述：
 * GeneralName结构用于多个扩展。该结构用于提供有关证书主体或颁发者的其他详细信息。
 *
 */
typedef enum {
    GENNAME_OTHNAME,
    GENNAME_RFC822NAME,
    GENNAME_DNSNAME,
    GENNAME_X400ADDR,
    GENNAME_DIRNAME,
    GENNAME_EDIPARTYNAME,
    GENNAME_URI,
    GENNAME_IPADDR,
    GENNAME_REGID
} X509ExtGenNameId;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief X509ExtGenName 结构体，用于传递 GeneralName 中的信息。
 *
 */
typedef struct {
    X509ExtGenNameId choiceId; /* < 可在实例中使用的选项 */

    union GenNameChoice {
        X509ExtOthName *othName;       /* <  OtherName结构是GeneralName的一部分，
                                        为最终实体证书的主题标记唯一标识符的其他名称 */
        Asn1IA5String *rfc822Name;     /* < rfc822 名称 */
        Asn1IA5String *dnsName;        /* < DNS 名称 */
        Asn1OctetString *x400Addr;     /* < x400 地址 */
        X509ExtName *dirName;          /* < x.500目录访问协议名称 */
        Asn1OctetString *ediPartyName; /* < 电子文档接口(EDI)信息如何交换信息和服务 */
        Asn1IA5String *uri;            /* < URL信息 */
        Asn1OctetString *ipAddr;       /* < IP地址 */
        Asn1Oid *regId;                /* < 从受信任的颁发机构获取的注册OID */
    } a;                               /* < 可以携带信息的联合体 */
} X509ExtGenName;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   创建带有 Directory name（X509ExtnRDNSequence类型）的 X509ExtGenName 结构。
 * @param   name [IN]  Directory name.
 * @retval  指向用Directory Name选项设置的GeneralName结构的指针。
 */
X509ExtGenName *X509EXT_GenNameCreateByName(const X509ExtName *name);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   创建 GeneralName。
 * @param   genNameType [IN] GeneralName类型。
 * @param   data [IN] 生成GeneralName的数据。
 * @param   len [IN] 数据长度。
 * @retval  X509ExtGenName，创建后的GeneralName，若失败，返回NULL。
 */
X509ExtGenName *X509EXT_GenNameCreate(X509ExtComType genNameType, const uint8_t *data, size_t len);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   比较两个GeneralName。
 * @param   srcGenName [IN] 第一个 GeneralName。
 * @param   destGenName [IN] 第二个 GeneralName。
 * @retval  true 对比一致。
 * @retval  false 对比不一致。
 */
bool X509EXT_GenNameCompare(const X509ExtGenName *srcGenName, const X509ExtGenName *destGenName);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   复制一份GeneralName。
 * @param   src [IN] 待复制的GeneralName。
 * @retval  已复制的GeneralName指针。
 */
X509ExtGenName *X509EXT_GenNameDump(const X509ExtGenName *src);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   复制一份 GeneralNameList。
 * @param   srcList [IN] 待复制的GeneralNameList。
 * @retval  已复制的 GeneralNameList 指针。
 */
X509ExtGenNameList *X509EXT_GenNameListDump(const X509ExtGenNameList *srcList);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   释放GeneralName。
 * @param   genName [IN] GeneralName。
 */
void X509EXT_GenNameFree(X509ExtGenName *genName);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   创建 GeneralName，所需信息包含在链表中。
 * @param   dataList [IN] 链表句柄，包含创建GeneralName需要的信息。
 * @retval  X509ExtGenName，创建后的GeneralName，若失败，返回NULL。
 * @par Note
 * 需要注意的是，CME支持的OID是有长度限制的.点分格式最多支持20个字段，二进制数据最多支持200个字节。
 */
X509ExtGenName *X509EXT_GenNameCreateByList(const X509ExtComDataList *dataList);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   释放GeneralName链表。
 * @param   genNameList [IN] GeneralName链表。
 */
void X509EXT_GenNameListFree(ListHandle genNameList);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief 表示各种字符串格式。
 *
 */
typedef enum {
    DIRSTR_TELETEXSTR, /* < 表示电传字符串格式 */
    DIRSTR_PRTABLESTR, /* < 表示可打印字符串格式 */
    DIRSTR_UNISTR,     /* < 表示通用字符串格式 */
    DIRSTR_UTF8STR,    /* < 表示UTF8字符串格式 */
    DIRSTR_BMPSTR,     /* < 表示BMP字符串格式 */
    DIRSTR_BUTT        /* < 枚举最大值 */
} X509ExtDirStrId;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief 在创建证书扩展时使用该结构。该结构支持不同的字符串格式。这些字符串格式都可以存在。
 */
typedef struct {
    X509ExtDirStrId choiceId; /* < 可以表示字符串的各种字符串格式 */
    union {
        Asn1TeletexString *teletexStr;   /* < Teletex格式 */
        Asn1PrintableString *prtableStr; /* < Printable格式 */
        Asn1UniversalString *uniStr;     /* < Universal格式 */
        Asn1UTF8String *utf8Str;         /* < UTF8格式 */
        Asn1BMPString *bmpStr;           /* < BMP格式 */
        Asn1OctetString *comStr;         /* < Octet格式 */
    } a;                                 /* < 通过联合来保存其中一个字符串表示形式 */
} X509ExtDirStr;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   从 X509ExtComData 结构列表创建 Subject Alternate Directory 列表。
 * @par 描述：
 * 这个API创建一个属性结构列表。这是标准支持的扩展。此扩展用于提供证书用户的附加标识。
 * 输入列表由 X509ExtComData 的列表组成结构。一些Subject Directory Attributes扩展中的值包括：
 * @li 出生地 [X509EXT_TYPE_PLACEOFBIRTH]。
 * @li 出生日期[X509EXT_TYPE_DOB]。
 * 出生日期必须是通用时间格式的字符串。使用SEC-DateTimeToGenTime将时间转换为通用时间。 输入列表必须是广义时间。
 * @param   dataList [IN] 包含创建 Subject Directory 属性列表的数据的列表。
 * @retval  成功执行后，将创建主题目录属性列表。
 */
X509ExtAttributeList *X509EXT_SubjectDirAttrCreate(const X509ExtComDataList *dataList);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   释放 Subject Directory Attribute。
 * @param   subDirAttrList [IN] 待释放的 Subject Directory Attribute 链表。
 */
void X509EXT_SubjectDirAttrFree(X509ExtAttributeList *subDirAttrList);

/**
 * X509ExtGenSubtree 结构体，与一个 subject name 关联，
 * 该 subject name 可以包含 iPAddress rfc822Name、directoryName等
 * 这些名称可以放在允许或排除列表中。
 * minimum 和 maximum 字段不与任何名称形式一起使用，因此 minimum 必须为零，maximum 必须不存在。
 */
typedef struct GeneralSubtree_ {
    X509ExtGenName *base; /* < 允许或排除列表中的名称。 */
    Asn1Int *minimum;     /* < 最小值为0，因为它未被使用 */
    Asn1Int *maximum;     /* < 通常没有最大值 */
} X509ExtGenSubtree;

/* List of subtrees */
typedef struct GeneralSubtree_ *X509ExtGenSubtreeHandle;

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   在给定 General Name 和其他必需字段的情况下，创建和填充 General SubTree 结构。
 * @param   generalName [IN] 必须包含在允许或排除列表中的名称。
 * @param   minimum [IN] 此值必须为0。
 * @param   maximum [IN] 通常不包含任何值。所以输入值必须始终为-1。
 * @retval  X509ExtGenSubtree * 指针。
 */
X509ExtGenSubtreeHandle X509EXT_GenSubTreeCreate(const X509ExtGenName *generalName, uint32_t minimum, int32_t maximum);

/**
 * @ingroup cme_x509v3_extn_dn
 * @brief   释放 GeneralSubtree 结构。
 * @param   genSubTree [IN] 待释放的 GeneralSubtree。
 */
void X509EXT_GenSubTreeFree(X509ExtGenSubtreeHandle genSubTree);

#ifdef __cplusplus
}
#endif

#endif // CME_X509V3_EXTN_DN_API_H
