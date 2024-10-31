/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: crypto algorithms id
 * Create: 2020/10/15
 * Notes:
 * History:
 * 2020/10/15 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_common CME通用接口
 *  @ingroup cme
 */
/** @defgroup cme_cid CME_CID枚举
 * @ingroup cme_common
 */
#ifndef CME_CID_API_H
#define CME_CID_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cme_cid
 * @brief crypto algorithms id
 * @attention 保存与OID对应的所有公共ID的枚举，新的算法ID应该添加到末尾。
 * 如果添加了新的对称加密算法，则 g_oidTable 数组应相应修改。
 */
typedef enum {
    CID_UNKNOWN, /* < 未知算法 */
    /* Algorithm Ids from crypto */
    CID_RC4,          /* < RC4算法 */
    CID_DES_ECB,      /* < ECB模式DES算法 */
    CID_DES_CBC,      /* < CBC模式DES算法 */
    CID_DES_OFB,      /* < OFB模式DES算法 */
    CID_DES_CFB,      /* < CFB模式DES算法 */
    CID_SCB2_128_ECB, /* < ECB模式SCB2-128算法 */
    CID_SCB2_128_CBC, /* < CBC模式SCB2-128算法 */
    CID_SCB2_256_ECB, /* < ECB模式SCB2-256算法 */
    CID_SCB2_256_CBC, /* < CBC模式SCB2-256算法 */

    CID_DES_EDE_ECB,  /* < ECB模式2个密钥三元组DES算法 */
    CID_DES_EDE_CBC,  /* < CBC模式2个密钥三元组DES算法 */
    CID_DES_EDE_OFB,  /* < OFB模式2个密钥三元组DES算法 */
    CID_DES_EDE_CFB,  /* < CFB模式2个密钥三元组DES算法 */
    CID_DES_EDE3_ECB, /* < ECB模式3个密钥三元组DES算法 */
    CID_DES_EDE3_CBC, /* < CBC模式3个密钥三元组DES算法 */
    CID_DES_EDE3_OFB, /* < OFB模式3个密钥三元组DES算法 */
    CID_DES_EDE3_CFB, /* < CFB模式3个密钥三元组DES算法 */
    CID_AES128_ECB,   /* < ECB模式AES-128算法 */
    CID_AES128_CBC,   /* < CBC模式AES-128算法 */
    CID_AES128_OFB,   /* < OFB模式AES-128算法 */
    CID_AES128_CFB,   /* < CFB模式AES-128算法 */
    CID_AES192_ECB,   /* < ECB模式AES-192算法 */
    CID_AES192_CBC,   /* < CBC模式AES-192算法 */
    CID_AES192_OFB,   /* < OFB模式AES-192算法 */
    CID_AES192_CFB,   /* < CFB模式AES-192算法 */
    CID_AES256_ECB,   /* < ECB模式AES-256算法 */
    CID_AES256_CBC,   /* < CBC模式AES-256算法 */
    CID_AES256_OFB,   /* < OFB模式AES-256算法 */
    CID_AES256_CFB,   /* < CFB模式AES-256算法 */
    CID_KASUMI_ECB,   /* < ECB模式Kasumi算法 */
    CID_KASUMI_CBC,   /* < CBC模式Kasumi算法 */
    CID_KASUMI_OFB,   /* < OFB模式Kasumi算法 */
    CID_KASUMI_CFB,   /* < CFB模式Kasumi算法 */
    CID_RSA,          /* < RSA算法 */
    CID_DSA,          /* < DSA算法 */
    CID_ECDSA,        /* < ECDSA算法 */
    CID_ECDSA192,     /* < ECDSA192算法 */
    CID_DH,           /* < Diffie-Hellman算法 */
    CID_ECDH,         /* < EC Diffie-Hellman算法 */
    CID_MD5,          /* < MD5 hash算法 */
    CID_SHA1,         /* < SHA1 hash算法 */
    CID_SHA224,       /* < SHA224 hash算法 */
    CID_SHA256,       /* < SHA256 hash算法 */
    CID_SHA384,       /* < SHA384 hash算法 */
    CID_SHA512,       /* < SHA512 hash算法 */
    CID_HMAC_MD5,     /* < MD5 hmac算法 */
    CID_HMAC_SHA1,    /* < SHA1 hmac算法 */
    CID_HMAC_SHA224,  /* < SHA224 hmac算法 */
    CID_HMAC_SHA256,  /* < SHA256 hmac算法 */
    CID_HMAC_SHA384,  /* < SHA384 hmac算法 */
    CID_HMAC_SHA512,  /* < SHA512 hmac算法 */
    CID_MD5WITHRSA,   /* < MD5和RSA签名算法 */
    CID_SHA1WITHRSA,  /* < SHA1和RSA签名算法 */
    /* identifies signature using SHA1 and RSA (coresponds to old Oid) */
    CID_SHA1WITHRSAOLD,     /* < SHA1和RSA签名算法(对应于旧Oid) */
    CID_DSAWITHSHA1,        /* < SHA1和DSA签名算法 */
    CID_DSAWITHSHA1_2,      /* < SHA1和DSA签名算法 */
    CID_ECDSAWITHSHA1,      /* < SHA1和ECDSA签名算法 */
    CID_ECDSAWITHSHA224,    /* < SHA224和ECDSA签名算法 */
    CID_ECDSAWITHSHA256,    /* < SHA256和ECDSA签名算法 */
    CID_ECDSAWITHSHA384,    /* < SHA384和ECDSA签名算法 */
    CID_ECDSAWITHSHA512,    /* < SHA512和ECDSA签名算法 */
    CID_ECDSA192WITHSHA256, /* < SHA256和ECDSA-192签名算法 */
    /* identifies signature using SHA256 and RSA */
    CID_SHA256WITHRSAENCRYPTION, /* < SHA256和RSA签名算法 */
    /* identifies signature using SHA384 and RSA */
    CID_SHA384WITHRSAENCRYPTION, /* < SHA384和RSA签名算法 */
    /* identifies signature using SHA512 and RSA */
    CID_SHA512WITHRSAENCRYPTION, /* < SHA512和RSA签名算法 */

    /* RFC 3279 */
    CID_KEYEXCHANGEALGORITHM,     /* < Key exchange算法标识符 */
    CID_PKCS1,                    /* < PKCS1 */
    CID_ANSI_X9_62,               /* < ANSI_X9_62 */
    CID_ECSIGTYPE,                /* < ECSIGTYPE */
    CID_FIELDTYPE,                /* < Field Type */
    CID_PRIME_FIELD,              /* < PRIME_FIELD */
    CID_CHARACTERISTIC_TWO_FIELD, /* < Characterstic Two field */
    CID_CHARACTERISTIC_TWO_BASIS, /* < Characterstic Two Basis */
    CID_GNBASIS,                  /* < GNBASIS */
    CID_TPBASIS,                  /* < TPBASIS */
    CID_PPBASIS,                  /* < PPBASIS */
    CID_PUBLICKEYTYPE,            /* < PUBLICKEYTYPE */
    CID_ELLIPTICCURVE,            /* < ELLIPTICCURVE */
    CID_C_TWOCURVE,               /* < C_TWOCURVE */
    CID_C2PNB163V1,               /* < C2PNB163V1 */
    CID_C2PNB163V2,               /* < C2PNB163V2 */
    CID_C2PNB163V3,               /* < C2PNB163V3 */
    CID_C2PNB176W1,               /* < C2PNB176W1 */
    CID_C2TNB191V1,               /* < C2TNB191V1 */
    CID_C2TNB191V2,               /* < C2TNB191V2 */
    CID_C2TNB191V3,               /* < C2TNB191V3 */
    CID_C2ONB191V4,               /* < C2ONB191V4 */
    CID_C2ONB191V5,               /* < C2ONB191V5 */
    CID_C2PNB208W1,               /* < C2PNB208W1 */
    CID_C2TNB239V1,               /* < C2TNB239V1 */
    CID_C2TNB239V2,               /* < C2TNB239V2 */
    CID_C2TNB239V3,               /* < C2TNB239V3 */
    CID_C2ONB239V4,               /* < C2ONB239V4 */
    CID_C2ONB239V5,               /* < C2ONB239V5 */
    CID_C2PNB272W1,               /* < C2PNB272W1 */
    CID_C2PNB304W1,               /* < C2PNB304W1 */
    CID_C2TNB359V1,               /* < C2TNB359V1 */
    CID_C2PNB368W1,               /* < C2PNB368W1 */
    CID_C2TNB431R1,               /* < C2TNB431R1 */
    CID_PRIMECURVE,               /* < PRIMECURVE */
    CID_PRIME192V1,               /* < PRIME192V1 */
    CID_PRIME192V2,               /* < PRIME192V2 */
    CID_PRIME192V3,               /* < PRIME192V3 */
    CID_PRIME239V1,               /* < PRIME239V1 */
    CID_PRIME239V2,               /* < PRIME239V2 */
    CID_PRIME239V3,               /* < PRIME239V3 */
    CID_PRIME256V1,               /* < PRIME256V1 */
    /* SCEP */
    CID_VERISIGN,       /* < VERISIGN */
    CID_PKI,            /* < PKI */
    CID_ATTRIBUTES,     /* < ATTRIBUTES */
    CID_MESSAGETYPE,    /* < MESSAGETYPE */
    CID_PKISTATUS,      /* < PKISTATUS */
    CID_FAILINFO,       /* < FAILINFO */
    CID_SENDERNONCE,    /* < SENDERNONCE */
    CID_RECIPIENTNONCE, /* < RECIPIENTNONCE */
    CID_TRANSID,        /* < TRANSID */
    CID_EXTENSIONREQ,   /* < EXTENSIONREQ */
    /* PKCS 5 */
    CID_RSADSI,              /* < RSADSI */
    CID_PKCS,                /* < PKCS */
    CID_PKCS5,               /* < PKCS5 */
    CID_PBKDF2,              /* < PBKDF2 */
    CID_RESERVED_1,          /* < RESERVED */
    CID_RESERVED_2,          /* < RESERVED */
    CID_PBE_MD5WITHDESCBC,   /* < PBE_MD5WITHDESCBC */
    CID_RESERVED_3,          /* < RESERVED */
    CID_PBE_SHA1WITHDESCBC,  /* < PBE_SHA1WITHDESCBC */
    CID_RESERVED_4,          /* < RESERVED */
    CID_PBES2,               /* < PBES2 */
    CID_PBMAC1,              /* < PBMAC1 */
    CID_DIGESTALGORITHM,     /* < DIGEST算法 */
    CID_ENCRYPTIONALGORITHM, /* < ENCRYPTION算法 */
    CID_RC2_CBC_OLD,         /* < RC2CBC */
    CID_RC5_CBC_PAD,         /* < RC5_CBC_PAD */
    CID_RSAES_OAEP,          /* < RSAES_OAEP */
    /* from pkcs1 */         /* identifies RSAES_OAEP */
    /* OCSP */
    CID_PKIX_OCSP_BASIC,           /* < OCSP_BASIC */
    CID_PKIX_OCSP_NONCE,           /* < OCSP_NONCE */
    CID_PKIX_OCSP_CRL,             /* < OCSP_CRL */
    CID_PKIX_OCSP_RESPONSE,        /* < OCSP_RESPONSE */
    CID_PKIX_OCSP_NOCHECK,         /* < OCSP_NOCHECK */
    CID_PKIX_OCSP_ARCHIVE_CUTOFF,  /* < OCSP_ARCHIVE_CUTOFF */
    CID_PKIX_OCSP_SERVICE_LOCATOR, /* < OCSP_SERVICE_LOCATOR */
    /* PKCS 10 */
    CID_CHALLENGE_PWD_ATTR, /* < Challenge PWD Attr */
    CID_EXTENSIONREQUEST,   /* < EXTENSIONREQUEST */
    /* FROM PKIXEXPLICIT */
    CID_PKIX,                      /* < PKIX */
    CID_PE,                        /* < PE */
    CID_QT,                        /* < QT */
    CID_KP,                        /* < KP */
    CID_AD,                        /* < AD */
    CID_QT_CPS,                    /* < CPS */
    CID_QT_UNOTICE,                /* < UNOTICE */
    CID_AD_OCSP,                   /* < OCSP */
    CID_AD_CAISSUERS,              /* < CAISSUERS */
    CID_AD_TIMESTAMPING,           /* < TIMESTAMPING */
    CID_AD_CAREPOSITORY,           /* < CAREPOSITORY */
    CID_AT,                        /* < AT */
    CID_AT_NAME,                   /* < NAME */
    CID_AT_SURNAME,                /* < SURNAME */
    CID_AT_GIVENNAME,              /* < GIVENNAME */
    CID_AT_INITIALS,               /* < INITIALS */
    CID_AT_GENERATIONQUALIFIER,    /* < GENERATIONQUALIFIER */
    CID_AT_COMMONNAME,             /* < COMMONNAME */
    CID_AT_LOCALITYNAME,           /* < LOCALITYNAME */
    CID_AT_STATEORPROVINCENAME,    /* < STATEORPROVINCENAME */
    CID_AT_ORGANIZATIONNAME,       /* < ORGANIZATIONNAME */
    CID_AT_ORGANIZATIONALUNITNAME, /* < ORGANIZATIONALUNITNAME */
    CID_AT_TITLE,                  /* < TITLE */
    CID_AT_DNQUALIFIER,            /* < DNQUALIFIER */
    CID_AT_COUNTRYNAME,            /* < COUNTRYNAME */
    CID_AT_SERIALNUMBER,           /* < SERIALNUMBER */
    CID_AT_PSEUDONYM,              /* < PSEUDONYM */
    CID_POSTAL_CODE,               /* < postal code */
    CID_DNS_NAME,                  /* < dns name */
    CID_USER_ID,                   /* < user id */
    CID_HOST,                      /* < host */
    CID_DOMAINCOMPONENT,           /* < DOMAINCOMPONENT */
    CID_EMAILADDRESS,              /* < EMAILADDRESS */
    /* PKIXIMPLICIT */
    CID_CE,                            /* < CE */
    CID_CE_AUTHORITYKEYIDENTIFIER,     /* < AUTHORITYKEYIDENTIFIER */
    CID_CE_SUBJECTKEYIDENTIFIER,       /* < SUBJECTKEYIDENTIFIER */
    CID_CE_KEYUSAGE,                   /* < KEYUSAGE */
    CID_CE_PRIVATEKEYUSAGEPERIOD,      /* < PRIVATEKEYUSAGEPERIOD */
    CID_CE_CERTIFICATEPOLICIES,        /* < CERTIFICATEPOLICIES */
    CID_ANYPOLICY,                     /* < ANYPOLICY */
    CID_CE_POLICYMAPPINGS,             /* < POLICYMAPPINGS */
    CID_CE_SUBJECTALTNAME,             /* < SUBJECTALTNAME */
    CID_CE_ISSUERALTNAME,              /* < ISSUERALTNAME */
    CID_CE_SUBJECTDIRECTORYATTRIBUTES, /* < SUBJECTDIRECTORYATTRIBUTES */
    CID_CE_BASICCONSTRAINTS,           /* < BASICCONSTRAINTS */
    CID_CE_NAMECONSTRAINTS,            /* < NAMECONSTRAINTS */
    CID_CE_POLICYCONSTRAINTS,          /* < POLICYCONSTRAINTS */
    CID_CE_CRLDISTRIBUTIONPOINTS,      /* < CRLDISTRIBUTIONPOINTS */
    CID_CE_EXTKEYUSAGE,                /* < EXTKEYUSAGE */
    CID_ANYEXTENDEDKEYUSAGE,           /* < ANYEXTENDEDKEYUSAGE */
    CID_KP_SERVERAUTH,                 /* < SERVERAUTH */
    CID_KP_CLIENTAUTH,                 /* < CLIENTAUTH */
    CID_KP_CODESIGNING,                /* < CODESIGNING */
    CID_KP_EMAILPROTECTION,            /* < EMAILPROTECTION */
    CID_KP_TIMESTAMPING,               /* < TIMESTAMPING */
    CID_KP_OCSPSIGNING,                /* < OCSPSIGNING */
    CID_KP_IPSECIKE,                   /* < IPSECIKE */
    CID_CE_INHIBITANYPOLICY,           /* < INHIBITANYPOLICY */
    CID_CE_FRESHESTCRL,                /* < FRESHESTCRL */
    CID_PE_AUTHORITYINFOACCESS,        /* < AUTHORITYINFOACCESS */
    CID_PE_SUBJECTINFOACCESS,          /* < SUBJECTINFOACCESS */
    CID_CE_CRLNUMBER,                  /* < CRLNUMBER */
    CID_CE_ISSUINGDISTRIBUTIONPOINT,   /* < ISSUINGDISTRIBUTIONPOINT */
    CID_CE_DELTACRLINDICATOR,          /* < DELTACRLINDICATOR */
    CID_CE_CRLREASONS,                 /* < CRLREASONS */
    CID_CE_CERTIFICATEISSUER,          /* < CERTIFICATEISSUER */
    CID_CE_HOLDINSTRUCTIONCODE,        /* < HOLDINSTRUCTIONCODE */
    CID_HOLDINSTRUCTION,               /* < HOLDINSTRUCTION */
    CID_HOLDINSTRUCTION_NONE,          /* < HOLDINSTRUCTION_NONE */
    CID_HOLDINSTRUCTION_CALLISSUER,    /* < HOLDINSTRUCTION_CALLISSUER */
    CID_HOLDINSTRUCTION_REJECT,        /* < HOLDINSTRUCTION_REJECT */
    CID_CE_INVALIDITYDATE,             /* < INVALIDITYDATE */
    CID_PDA_DATEOFBIRTH,               /* < DATEOFBIRTH */
    CID_PDA_PLACEOFBIRTH,              /* < PLACEOFBIRTH */
    CID_PDA_GENDER,                    /* < GENDER */
    CID_PDA_COUNTRYOFCITIZENSHIP,      /* < COUNTRYOFCITIZENSHIP */
    CID_PDA_COUNTRYOFRESIDENCE,        /* < COUNTRYOFRESIDENCE */
    CID_PDA,                           /* < PDA */
    CID_ON_PERMANENTIDENTIFIER,        /* < PERMANENTIDENTIFIER */
    CID_ON,                            /* < ON */
    CID_CE_DOMAININFO,                 /* < DOMAININFO */
    /* CMP */
    CID_PASSWORDBASEDMAC, /* < PWD Based MAC */
    CID_DHBASEDMAC,       /* < DH Based MAC */
    CID_IT,               /* < IT */
    CID_CAPROTENCCERT,    /* < CAPROTENCCERT */
    CID_SIGNKEYPAIRTYPES, /* < Sign KeyPair Types */
    CID_ENCKEYPAIRTYPES,  /* < KeyPair Types */
    CID_PREFERREDSYMMALG, /* < Preferred Symmetric Algo */
    CID_CAKEYUPDATEINFO,  /* < CA Key Update Info */
    CID_CURRENTCRL,       /* < Current CRL */
    CID_CONFIRMWAITTIME,  /* < ConfirmWaitTime */
    /* CRMF */
    CID_PKIP,                       /* < PKIP */
    CID_REGCTRL,                    /* < REGCTRL */
    CID_REGCTRL_REGTOKEN,           /* < REGTOKEN */
    CID_REGCTRL_AUTHENTICATOR,      /* < AUTHENTICATOR */
    CID_REGCTRL_PKIPUBLICATIONINFO, /* < PKIPUBLICATIONINFO */
    CID_REGCTRL_PKIARCHIVEOPTIONS,  /* < PKIARCHIVEOPTIONS */
    CID_REGCTRL_OLDCERTID,          /* < OLDCERTID */
    CID_REGCTRL_PROTOCOLENCRKEY,    /* < PROTOCOLENCRKEY */
    CID_REGINFO,                    /* < REGINFO */
    CID_REGINFO_UTF8PAIRS,          /* < UTF8PAIRS */
    CID_REGINFO_CERTREQ,            /* < CERTREQ */
    /* PKCS12 */
    CID_PKCS12,               /* < PKCS12 */
    CID_PKCS12PBEIDS,         /* < PKCS12 PBE */
    CID_PBE_SHAWITH128BITRC4, /* < PBE Algo (SHAWITH128BITRC4) */
    CID_PBE_SHAWITH40BITRC4,  /* < PBE Algo (SHAWITH40BITRC4) */
    /* identifies PBE Algo (SHAWITH3KEY_TRIPLE_DESCBC) */
    CID_PBE_SHAWITH3KEY_TRIPLE_DESCBC, /* < PBE Algo (SHAWITH3KEY_TRIPLE_DESCBC) */
    /* identifies PBE Algo (SHAWITH2KEY_TRIPLE_DESCBC) */
    CID_PBE_SHAWITH2KEY_TRIPLE_DESCBC, /* < PBE Algo (SHAWITH2KEY_TRIPLE_DESCBC) */

    /* identifies PBE Algo (SHAWITH128BIT_RC2CBC) */
    CID_PBE_SHAWITH128BIT_RC2CBC, /* < PBE Algo (SHAWITH128BIT_RC2CBC) */
    CID_PBE_SHAWITH40BIT_RC2CBC,  /* < PBE Algo (SHAWITH40BIT_RC2CBC) */
    CID_BAGTYPES,                 /* < Bag Types */
    CID_KEYBAG,                   /* < Key Bag */
    CID_PKCS8SHROUDEDKEYBAG,      /* < Bag Types */
    CID_CERTBAG,                  /* < CERT Bag */
    CID_CRLBAG,                   /* < CRL Bag */
    CID_SECRETBAG,                /* < Secret Bag */
    CID_SAFECONTENTSBAG,          /* < Safe Content Bag */
    CID_X509CERTIFICATE,          /* < x509 Certificate */
    CID_SDSICERTIFICATE,          /* < SDSI Certificate */
    CID_FRIENDLYNAME,             /* < Freidnly Name */
    CID_LOCALKEYID,               /* < Local Key ID */
    /* auth_frame */
    CID_CERTIFICATEREVOCATIONLIST, /* < Certificate Revocation List */
    /* PKCS7 & 9 */
    CID_PKCS7,                      /* < PKCS7 */
    CID_PKCS7_SIMPLEDATA,           /* < PKCS7 Simple Data */
    CID_PKCS7_SIGNEDDATA,           /* < PKCS7 Signed Data */
    CID_PKCS7_ENVELOPEDDATA,        /* < PKCS7 Enveloped Data */
    CID_PKCS7_SIGNED_ENVELOPEDDATA, /* < PKCS7 Signed Enveloped Data */
    CID_PKCS7_DIGESTEDDATA,         /* < PKCS7 Degested Data */
    CID_PKCS7_ENCRYPTEDDATA,        /* < PKCS7 Encrypted Data */
    CID_PKCS9,                      /* < PKCS9 */
    CID_PKCS9_AT_CONTENTTYPE,       /* < PKCS9 Content Type */
    CID_PKCS9_AT_MESSAGEDIGEST,     /* < PKCS9 Message Digest */
    CID_PKCS9_AT_SIGNINGTIME,       /* < PKCS9 Signing time */
    CID_PKCS9_AT_COUNTERSIGNATURE,  /* < PKCS9 Counter Signature */
    CID_PKCS9_AT_RANDOMNONCE,       /* < PKCS9 Signed Enveloped Data */
    CID_PKCS9_AT_SEQUENCENUMBER,    /* < PKCS9 Sequence number */
    /* Additional Algorithms provided by crypto */
    CID_MD4,       /* < MD4 hash算法 */
    CID_HMAC_MD4,  /* < 使用MD4的hmac */
    CID_CMAC_AES,  /* < CMAC-AES */
    CID_CMAC_TDES, /* < CMAC-Triple DES */
    CID_RNG_HW,    /* < TRNG */
    CID_RNG_SW,    /* < PRNG */
    CID_XCBC_AES,  /* < XCBC-MAC-AES */
    CID_RC2_ECB,   /* < ECB模式RC2算法 */
    CID_RC2_CBC,   /* < CBC模式RC2算法 */
    CID_RC2_OFB,   /* < OFB模式RC2算法 */
    CID_RC2_CFB,   /* < CFB模式RC2算法 */
    CID_MD5_SHA1,  /* < CFB模式RC2算法 */

    CID_SECP384R1,         /* < NIST prime curve 384 */
    CID_SECP521R1,         /* < NIST prime curve 521 */
    CID_SM3,               /* < SM3 hash算法 */
    CID_HMAC_SM3,          /* < hmac with SM3 */
    CID_SM2DSAWITHSM3,     /* < SM2DSAWITHSM3 */
    CID_SM2DSAWITHSHA1,    /* < SM2DSAWITHSHA1 */
    CID_SM2DSAWITHSHA256,  /* < SM2DSAWITHSHA256 */
    CID_SM2PRIME256,       /* < PRIME256SM2 */
    CID_SM2DSA,            /* < SM2 DSA */
    CID_SM2KEP,            /* < SM2KEP */
    CID_SM2PKEA,           /* < SM2PKEA */
    CID_AES128_GCM,        /* < GCM模式AES128算法 */
    CID_AES192_GCM,        /* < GCM模式AES192算法 */
    CID_AES256_GCM,        /* < GCM模式AES256算法 */
    CID_AES128_CTR,        /* < CTR模式AES128算法 */
    CID_AES192_CTR,        /* < CTR模式AES192算法 */
    CID_AES256_CTR,        /* < CTR模式AES256算法 */
    CID_UNSTRUCTURED_NAME, /* < unstructured name */
    CID_UNSTRUCTURED_ADDR, /* < unstructuredAddress */
    CID_BF_ECB,            /* < ECB模式 Blowfish算法 */
    CID_BF_CBC,            /* < CBC模式 Blowfish算法 */
    CID_BF_CFB,            /* < OFB模式Blowfish算法 */
    CID_BF_OFB,            /* < in OFB模式Blowfish算法 */
    CID_AES128_CCM,        /* < CCM模式AES128算法 */
    CID_AES192_CCM,        /* < CCM模式AES192算法 */
    CID_AES256_CCM,        /* < CCM模式AES256算法 */

    CID_AT_STREETADDRESS,       /* <  EV certs中的streetAddress */
    CID_AT_BUSINESSCATEGORY,    /* < EV certs中的businessCategory in EV certs */
    CID_AT_POSTALCODE,          /* < EV certs中的postalCode */
    CID_JD_LOCALITYNAME,        /* < EV certs中的streetAddress */
    CID_JD_STATEORPROVINCENAME, /* < EV certs中的jurisdictionLocalityName */
    CID_JD_COUNTRYNAME,         /* < EV certs中的jurisdictionCountryName */
    CID_HMAC_SHA1_DIGEST,

    CID_NIST_PRIME224,                /* < NIST Curve P-224 */
    CID_NIST_C2PNB163K,               /* < NIST 二进制 Curve 163K */
    CID_NIST_C2PNB163B,               /* < NIST 二进制 Curve 163B */
    CID_NIST_C2TNB233K,               /* < NIST 二进制 Curve 233K */
    CID_NIST_C2TNB233B,               /* < NIST 二进制 Curve 233B */
    CID_NIST_C2PNB283K,               /* < NIST 二进制 Curve 283K */
    CID_NIST_C2PNB283B,               /* < NIST 二进制 Curve 283B */
    CID_NIST_C2TNB409K,               /* < NIST 二进制 Curve 409K */
    CID_NIST_C2TNB409B,               /* < NIST 二进制 Curve 409B */
    CID_NIST_C2PNB571K,               /* < NIST 二进制 Curve 571K */
    CID_NIST_C2PNB571B,               /* < NIST 二进制 Curve 571B */
    CID_PBE_HMACSHA512WITHAES256_CBC, /* < NIST 二进制 Curve 571B */

    CID_CE_SKAE, /* < SKAE extension */
    CID_ED25519, /* < ED25519算法 */
    CID_SM4_BLOCK_CIPHER,              /* < SM4 block cipher 算法 */
    CID_SM2_PUBLIC_KEY_ENCRYPTION_21, /* SM2 公钥加密 */
    CID_BUTT     /* < CID结束值 */
} CmeCid;

#ifdef __cplusplus
}
#endif

#endif // CME_CID_API_H
