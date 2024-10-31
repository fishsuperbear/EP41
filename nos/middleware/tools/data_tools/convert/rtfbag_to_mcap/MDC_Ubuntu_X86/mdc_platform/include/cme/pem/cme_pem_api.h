/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 包含PEM格式的编解码功能
 * Create: 2020/11/07
 * Notes:
 * History:
 * 2020/11/07 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_x509 CME_X509_API
 * @ingroup cme
 */
/** @defgroup cme_pem CME_PEM_API
 * @ingroup cme_x509
 */
#ifndef CME_PEM_API_H
#define CME_PEM_API_H

#include "cme_cid_api.h"
#include "x509/cme_x509_crl_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_pem
 * @brief 表示各种pem对象类型。
 * @attention 所述对象类型用于标识PEM_Encode / PEM_Decode的输入/输出。
 */
typedef enum {
    PEM_OBJTYPE_PKCS10_REQUEST = 0,  /* < PKCS10Req */
    PEM_OBJTYPE_CERT = 1,            /* < X509Cert */
    PEM_OBJTYPE_CRL = 2,             /* < X509CRLCertList */
    PEM_OBJTYPE_OCSP_REQUEST = 3,    /* < CmeOcspRequest */
    PEM_OBJTYPE_OCSP_RESPONSE = 4,   /* < CmeOcspResp */
    PEM_OBJTYPE_PRIVATEKEY = 5,      /* < PKeyAsymmetricKey Private Key */
    PEM_OBJTYPE_PUBLICKEY = 6,       /* < PKeyAsymmetricKey Public Key */
    PEM_OBJTYPE_PKCS7_ENVELOPED = 7, /* < PKCS7 Enveloped Data */
    PEM_OBJTYPE_UNKNOWN = 8          /* < UNKNOW */
} PemObjType;

/**
 * @ingroup cme_pem
 * @brief 不同proc类型的表示（process type参考RFC 1421）
 * @attention 输入对象可以以2种格式编码：
 * @li MIC_ONLY，数据的完整性受到保护，数据被编码为base 64格式。
 * @li ENCRYPTED，数据的完整性受到保护，数据被加密。
 * @par 注释：
 * MIC为消息完整性码。
 */
typedef enum {
    MIC_CLEAR = 0, /* < 数据完整性保护，数据不能被修改。PEM任何对象类型都不应该支持此操作。 */
    MIC_ONLY = 1, /* < 数据的完整性受到保护，数据被编码为基64格式。 */
    ENCRYPTED = 2 /* < 数据的完整性受到保护，数据被加密 */
} PemProcType;

/**
 * @ingroup cme_pem
 * @brief 将给定数据编码为PEM格式。
 *
 * @par 目的：
 * 将给定数据编码为PEM格式。
 *
 * @par 描述：
 * 根据选项，该函数将输入缓冲区中的对象编码为PEM格式。这里的对象由任何一种PemObjType表示。\n
 * 如果proc类型为MIC_ONLY，则enAlgId、encKey、ulKeyLen可以赋值为NULL。\n
 * 如果proc类型是ENCRYPTED，则必须给出enAlgId、encKey、ulKeyLen的有效值。\n
 * 有关PEM的详细信息在4个RFC中记录： 1421, 1422, 1423和1424。
 *
 * @param[IN] object 以PEM格式编码的对象。\n
 * 该对象可以是PemObjType中提到的任何对象。 例如: X509 Certificate,CRL,OCSP Request,OCSP Response。\n
 * Object必须是enObjectType参数中提到的结构体。\n
 * 例如, 如果enObjectType = PEM_OBJTYPE_CERT，那么对应的是证书类型的编码，
 * 其他对象类型应该根据PemObjType所包含的类型来进行相应对象类型的编码。
 * @param [IN] enObjectType 对象类型：指定pObject的类型。例如：PEM_OBJTYPE_CERT。
 * @param [IN] enProcType Proc 类型，创建PEM消息的过程。可能值为MIC_ONLY、ENCRYPTED。
 * @param [IN] enAlgId 对称加密算法id。(当前版本无效)
 * @param [IN] encKey 用于创建对称加密密钥的密码。(当前版本无效)
 * @param [IN] keyLen 给定密码的长度。应该在[1, 0xffff]范围内。(当前版本无效)
 *
 * @retval char* 成功完成时，PEM格式化输出数据。
 * @retval NULL 出错时，返回NULL。
 *
 * @par 依赖：
 * pem.h
 *
 * @par 内存处理：
 * @li 应调用ipsi_free函数释放来自PEM_Encode函数的编码输出缓冲区。
 * @li PEM_Encode返回的内存中含有'\0'结束符，用户应将该内存原样传递给解码API（包含'\0'结束符）。
 * 如果将PEM_Encode返回的内存传递给以缓冲区和缓冲区长度作为输入的API，则传递的缓冲区长度还应包括'\0'结束符。
 * 由strlen计算的缓冲区长度错误的，因为它不包括'\0'结束符。
 */
char *PEM_Encode(const void *object, PemObjType enObjectType, PemProcType enProcType, CmeCid enAlgId,
                 const char *encKey, size_t keyLen);

/**
 * @ingroup cme_pem
 * @brief 将编码的PEM对象解码成其原始对象类型。
 *
 * @par 目的：
 * 将编码的PEM对象解码成其原始对象类型。
 *
 * @par 描述：
 * 此函数将输入PEM格式解码为所需对象。此API将解码以PEM编码的输入缓冲区数据。\n
 * 可以包含以下任何一项：\n
 * @li Certificate,CRL.
 * @li OCSP request.
 * @li OCSP response.
 * @li PKCS 7, 8, 10 messages.
 * 对象类型输入参数将标识所需的输出对象。此接口将获取输入的PEM格式数据，
 * 如果输入参数错误，则返回请求的对象，并返回null。\n
 * 有关PEM的详细信息在4个RFC中记录： 1421、1422、1423和1424。
 *
 * @param [IN] pemBuf 输入PEM缓冲区必须为空终止缓冲区。
 * @param [IN] objType 对象类型，指定pObject的类型。例如，PEM_OBJTYPE_CERT。
 * @param [IN] encKey 对称加密密钥。
 * @param [IN] keyLen 对称加密密钥长度。应该在[1, 0xffff]。
 *
 * @retval void* 成功，PEM解码后的输出对象。
 * 该对象可以是PemObjType中提到的任何对象。例如， X509证书、CRL、OCSP请求、OCSP响应等。
 * @retval void* 错误，返回NULL。
 *
 * @par 依赖：
 * pem.h.
 *
 * @par 注释：
 * @li 该API操作pucPEMBuf。因此，pucPEMBuf不应该是只读缓冲区。
 * @li 此API的输入缓冲区（pucPEMBuf）的最大字符串长度应小于500MB。
 * @li “------END”后的最大连续空格应小于8K，否则会导致失败。
 * @li 结束标签后的最大连续\r、\n或\r\n应小于8K，否则会导致失败。
 *
 * @par 内存操作：
 * 应根据解码后的对象类型释放来自PEM_Decode函数的解码输出缓冲区：\n
 * @li 对于PEM_OBJTYPE_CERT，使用X509_FreeCert。
 * @li 对于PEM_OBJTYPE_CRL，应使用X509CRL_Free。
 * @li 对于PEM_OBJTYPE_PKCS10_REQUEST，应使用PKCS10_FreeCertReq
 * @li 对于PEM_OBJTYPE_OCSP_REQUEST，应使用OCSP_FreeOCSPReq。
 * @li 对于PEM_OBJTYPE_OCSP_RESPONSE，应使用OCSP_freeOCSPResp。
 * @li 对于PEM_OBJTYPE_PRIVATEKEY，应使用PKEY_FreeKey。
 */
void *PEM_Decode(const char *pemBuf, PemObjType objType, const char *encKey, size_t keyLen);

/**
 * @ingroup cme_pem
 * @brief 将证书列表以PEM格式编码。
 *
 * @par 目的：
 * 将证书列表以PEM格式编码。
 *
 * @par 描述：
 * 对作为输入参数传递的证书列表进行编码。
 *
 * @param [IN] certList 持有X509证书的列表。
 *
 * @retval char* 成功，输出PEM格式编码证书列表。
 * @retval char* 失败，返回NULL。
 *
 * @par 依赖：
 * pem.h.
 *
 * @par 内存操作：
 * @li 应调用ipsi_free函数释放来自pem_encodeCertList函数的编码输出缓冲区。
 * @li PEM_Encode返回的内存中含有'\0'结束符，用户应将该内存原样传递给解码API（包含'\0'结束符）。
 * 如果将PEM_Encode返回的内存传递给以缓冲区和缓冲区长度作为输入的API，则传递的缓冲区长度还应包括'\0'结束符。
 * 由strlen计算的缓冲区长度错误的，因为它不包括'\0'结束符。
 */
char *PEM_CertListEncode(ListRoHandle certList);

/**
 * @ingroup cme_pem
 * @brief 解码编码的PEM格式证书列表。
 * @par 目的：
 * 解码编码的PEM格式证书列表。
 *
 * @par 描述：
 * 解码以PEM格式编码的证书列表，并作为入参传入。
 *
 * @param [IN] certListData PEM编码的证书缓冲区列表。pcEncodedCertList必须是空终止缓冲区。
 *
 * @retval CmeList* 成功，返回包含X509证书的列表。
 * @retval CmeList* 失败，返回NULL。
 *
 * @par 依赖：
 * pem.h.
 *
 * @par 注释：
 * @li 此接口输入缓冲区（certListData）的最大字符串长度应小于500MB。
 * @li 结束标签后的最大连续\r、\n或\r\n应小于8K，否则会导致失败。
 *
 * @par 内存操作：
 * PEM_DecodeCertList解码输出的证书列表应该使用LIST_FREE,即使用类型为freeFunc的X509_FreeCert。
 */
ListHandle PEM_CertListDecode(const char *certListData);

/**
 * @ingroup cme_pem
 * @brief 设置PEM certificates & CRLs的缓冲区大小限制。
 *
 * @par 目的：
 * 设置PEM certificates & CRLs的缓冲区大小限制（以字节为单位）。
 *
 * @par 描述：
 * 设置PEM certificates & CRLs的缓冲区大小限制（以字节为单位）。由于PEM certificates & CRLs可能很大，
 * 用户可以设置他们希望在其应用程序中允许的最大限制。如果不设置，默认为256K (0x40000)字节。
 * @param [IN] uiSize Maximum size upto 500 MB, which application wants to set [N/A]
 *
 * @retval CME_SUCCESS 成功。
 * @retval CME_ERR_BAD_PARAM 参数错误。
 *
 * @par 依赖：
 * pem.h.
 *
 * @par 注释：
 * @li 如果应用程序传递大小小于256K (0x40000)字节，则库仍将大小限制设置为256K (0x40000)字节，以确保向后兼容性。
 * @li 上限设置为500M (0x1F400000)字节。
 * @li 如果有非常大的证书和CRL，用户需要使PEM缓冲区是以'\0'终止
 * (应用程序在调用API加载PEM文件/缓冲区之前，可以判断buf[len - 1] == '\0')。
 * 如果没有终止NULL，库将尝试为整个缓冲区分配内存，然后附加"\0"。
 * 但如果缓冲区长度非常大（很可能是无效的长度），系统malloc不能分配这么多内存，这可能会导致未定义的行为。
 * @li 不应该在多个线程中调用此函数。此函数不是多线程安全函数。
 * @li 此API应该在init之后调用。不应与证书加载并行调用。
 */
int32_t PEM_PemBufMaxSizeSet(uint32_t maxValue);

/**
 * @ingroup cme_pem
 * @brief 获取PEM certificates & CRLs的缓冲区大小限制。
 *
 * @retval uint32_t 获取到的缓冲区大小限制值。
 */
size_t PEM_PemBufMaxSizeGet(void);

/**
 * @ingroup cme_pem
 * @brief   根据输入的PEM对象内容，获取其对应的PemObjTypeEnum类型。
 * @param   pemBuf [IN] PEM格式的数据内容。
 * @retval  PemObjType 例如PEM_OBJTYPE_CRL、PEM_OBJTYPE_CERT等表示为CRL、证书等。
 */
PemObjType PEM_PemObjTypeValueGet(const char *pemBuf);

/**
 * @ingroup cme_pem
 * @brief   根据输入的多个PEM编码的CERT数据及其指定长度, 解码为CERT对象链表。
 * @param   certListData [IN] PEM格式的数据内容。
 * @param   bufLen [IN] pemBuf的有效长度。
 * @retval  ListHandle 存储多个X509Cert数据的链表。
 */
ListHandle PEM_CertListDecodeWithLen(const char *certListData, size_t bufLen);

/**
 * @ingroup cme_pem
 * @brief 根据输入的多个PEM编码的CRL数据及其指定长度, 解码为X509_CRL_S对象。
 * @param encodedCrl [IN] PEM格式的数据内容。
 * @param crlLen [IN] PEM格式的数据内容长度。
 * @param encKey [IN] 解密key。
 * @param keyLen [IN] 解密key长度。
 * @retval X509CRLCertList* X509_CRL_S对象指针。
 */
X509CRLCertList *PEM_CrlDecodeWithLen(const char *encodedCrl, size_t crlLen, const char *encKey, size_t keyLen);

/**
 * @ingroup cme_pem
 * @brief 根据输入的PEM编码内容、编码类型、及其指定长度进行解码
 * @param encodedBuf [IN] PEM格式的数据内容
 * @param bufLen [IN] PEM格式的数据内容长度
 * @param objType [IN] 编码内容的类型
 * @param encKey [IN] 解密key
 * @param keyLen [IN] 解密key长度
 * @retval void* 解密后的数据流
 */
void *PEM_DecodeWithLen(const char *encodedBuf, size_t bufLen, PemObjType objType, const char *encKey, size_t keyLen);

/**
 * @ingroup cme_pem
 * @brief 用于编码证书的PEM编码宏。对于proctype MIC_ONLY,enAlgId、encKey、ulKeyLen可以为空。
 *
 * @param [IN] cert CERT类型。
 * @param [IN] enProcType Proc 类型，创建PEM消息的过程。可能值为MIC_ONLY、ENCRYPTED。
 * @param [IN] enAlgId 对称加密算法id。
 * @param [IN] encKey 用于创建对称加密密钥的密码。
 * @param [IN] keyLen 给定密码的长度。应该在[1, 0xffff]范围内。
 *
 * @retval char* 成功完成时，PEM格式化输出数据。
 * @retval NULL 出错时，返回NULL。
 */
#define PEM_CERT_ENCODE(cert, enProcType, enAlgId, encKey, keyLen) \
    PEM_Encode(cert, PEM_OBJTYPE_CERT, enProcType, enAlgId, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief PEM_OBJTYPE_CERT（对象类型证书的PEM解码宏）。
 *
 * @param [IN] pemText 输入PEM缓冲区必须为空终止缓冲区。
 * @param [IN] encKey 对称加密密钥。
 * @param [IN] keyLen 对称加密密钥长度。应该在[1, 0xffff]。
 *
 * @retval void* 成功，PEM解码后的输出对象。
 * @retval void* 错误，返回NULL。
 */
#define PEM_CERT_DECODE(pemText, encKey, keyLen) PEM_Decode(pemText, PEM_OBJTYPE_CERT, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief PEM编码宏，用于编码CRL。对于proctype MIC_ONLY,enAlgId、encKey、ulKeyLen可以为空。
 *
 * @param [IN] crl CRL类型。
 * @param [IN] enProcType Proc 类型，创建PEM消息的过程。可能值为MIC_ONLY、ENCRYPTED。
 * @param [IN] enAlgId 对称加密算法id。
 * @param [IN] encKey 用于创建对称加密密钥的密码。
 * @param [IN] keyLen 给定密码的长度。应该在[1, 0xffff]范围内。
 *
 * @retval char* 成功完成时，PEM格式化输出数据。
 * @retval NULL 出错时，返回NULL。
 */
#define PEM_CRL_ENCODE(crl, enProcType, enAlgId, encKey, keyLen) \
    PEM_Encode(crl, PEM_OBJTYPE_CRL, enProcType, enAlgId, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief PEM_OBJTYPE_CRL（对象类型CRL的PEM解码宏）。
 *
 * @param [IN] pemText 输入PEM缓冲区必须为空终止缓冲区。
 * @param [IN] encKey 对称加密密钥。
 * @param [IN] keyLen 对称加密密钥长度。应该在[1, 0xffff]。
 *
 * @retval void* 成功，PEM解码后的输出对象。
 * @retval void* 错误，返回NULL。
 */
#define PEM_CRL_DECODE(pemText, encKey, keyLen) PEM_Decode(pemText, PEM_OBJTYPE_CRL, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief PEM编码宏，用于编码PKCS10请求。对于proctype MIC_ONLY,enAlgId、encKey、ulKeyLen可以为空。
 *
 * @param [IN] pP10Req PKCS10类型。
 * @param [IN] enProcType Proc 类型，创建PEM消息的过程。可能值为MIC_ONLY、ENCRYPTED。
 * @param [IN] enAlgId 对称加密算法id。
 * @param [IN] encKey 用于创建对称加密密钥的密码。
 * @param [IN] keyLen 给定密码的长度。应该在[1, 0xffff]范围内。
 *
 * @retval char* 成功完成时，PEM格式化输出数据。
 * @retval NULL 出错时，返回NULL。
 */
#define PEM_PKCS10REQ_ENCODE(p10Req, enProcType, enAlgId, encKey, keyLen) \
    PEM_Encode(p10Req, PEM_OBJTYPE_PKCS10_REQUEST, enProcType, enAlgId, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief PEM解码对象类型PKCS10请求的宏(PEM_OBJTYPE_PKCS10_REQUEST)。
 *
 * @param [IN] pemText 输入PEM缓冲区必须为空终止缓冲区。
 * @param [IN] encKey 对称加密密钥。
 * @param [IN] keyLen 对称加密密钥长度。应该在[1, 0xffff]。
 *
 * @retval void* 成功，PEM解码后的输出对象。
 * @retval void* 错误，返回NULL。
 */
#define PEM_PKCS10REQ_DECODE(pemText, encKey, keyLen) PEM_Decode(pemText, PEM_OBJTYPE_PKCS10_REQUEST, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief 用于编码OCSP请求的PEM编码宏。对于proctype MIC_ONLY,enAlgId、encKey、ulKeyLen可以为空。
 *
 * @param [IN] ocspReq OCSP类型。
 * @param [IN] enProcType Proc 类型，创建PEM消息的过程。可能值为MIC_ONLY、ENCRYPTED。
 * @param [IN] enAlgId 对称加密算法id。
 * @param [IN] encKey 用于创建对称加密密钥的密码。
 * @param [IN] keyLen 给定密码的长度。应该在[1, 0xffff]范围内。
 *
 * @retval char* 成功完成时，PEM格式化输出数据。
 * @retval NULL 出错时，返回NULL。
 */
#define PEM_OCSPREQ_ENCODE(ocspReq, enProcType, enAlgId, encKey, keyLen) \
    PEM_Encode(ocspReq, PEM_OBJTYPE_OCSP_REQUEST, enProcType, enAlgId, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief 对象类型OCSP请求的PEM解码宏(PEM_OBJTYPE_OCSP_REQUEST)。
 *
 * @param [IN] pemText 输入PEM缓冲区必须为空终止缓冲区。
 * @param [IN] encKey 对称加密密钥。
 * @param [IN] keyLen 对称加密密钥长度。应该在[1, 0xffff]。
 *
 * @retval void* 成功，PEM解码后的输出对象。
 * @retval void* 错误，返回NULL。
 */
#define PEM_OCSPREQ_DECODE(pemText, encKey, keyLen) PEM_Decode(pemText, PEM_OBJTYPE_OCSP_REQUEST, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief 用于编码OCSP response的PEM编码宏。对于proctype MIC_ONLY,enAlgId、encKey、ulKeyLen可以为空。
 *
 * @param [IN] ocspResp OCSP response类型。
 * @param [IN] enProcType Proc 类型，创建PEM消息的过程。可能值为MIC_ONLY、ENCRYPTED。
 * @param [IN] enAlgId 对称加密算法id。
 * @param [IN] encKey 用于创建对称加密密钥的密码。
 * @param [IN] keyLen 给定密码的长度。应该在[1, 0xffff]范围内。
 *
 * @retval char* 成功完成时，PEM格式化输出数据。
 * @retval NULL 出错时，返回NULL。
 */
#define PEM_OCSPRESP_ENCODE(ocspResp, enProcType, enAlgId, encKey, keyLen) \
    PEM_Encode(ocspResp, PEM_OBJTYPE_OCSP_RESPONSE, enProcType, enAlgId, encKey, keyLen)

/**
 * @ingroup cme_pem
 * @brief 对象类型OCSP response的PEM解码宏(PEM_OBJTYPE_OCSP_RESPONSE)。
 *
 * @param [IN] pemText 输入PEM缓冲区必须为空终止缓冲区。
 * @param [IN] encKey 对称加密密钥。
 * @param [IN] keyLen 对称加密密钥长度。应该在[1, 0xffff]。
 *
 * @retval void* 成功，PEM解码后的输出对象。
 * @retval void* 错误，返回NULL。
 */
#define PEM_OCSPRESP_DECODE(pemText, encKey, keyLen) PEM_Decode(pemText, PEM_OBJTYPE_OCSP_RESPONSE, encKey, keyLen)

#ifdef __cplusplus
}
#endif /* end of __cplusplus */

#endif
