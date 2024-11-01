/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: EC注册接口对外头文件
 * Create: 2020/05/18
 * History:
 */
#ifndef CRYPTO_CRYPTO_ECC_API_H
#define CRYPTO_CRYPTO_ECC_API_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup crypto
 * crypto 椭圆曲线枚举
 */
typedef enum {
    CRYPTO_EC_GROUPID_X9_62_C2PNB163V1, /* < 椭圆曲线：C2PNB163V1 */
    CRYPTO_EC_GROUPID_X9_62_C2PNB163V2, /* < 椭圆曲线：C2PNB163V2 */
    CRYPTO_EC_GROUPID_X9_62_C2PNB163V3, /* < 椭圆曲线：C2PNB163V3 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB191V1, /* < 椭圆曲线：C2PNB191V1 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB191V2, /* < 椭圆曲线：C2PNB191V2 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB191V3, /* < 椭圆曲线：C2PNB191V3 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB239V1, /* < 椭圆曲线：C2TNB239V1 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB239V2, /* < 椭圆曲线：C2TNB239V2 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB239V3, /* < 椭圆曲线：C2TNB239V3 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB359V1, /* < 椭圆曲线：C2TNB359V1 */
    CRYPTO_EC_GROUPID_X9_62_C2TNB431R1, /* < 椭圆曲线：C2TNB431V1 */

    CRYPTO_EC_GROUPID_X9_62_PRIME192V2, /* < 椭圆曲线：PRIME192V2 */
    CRYPTO_EC_GROUPID_X9_62_PRIME192V3, /* < 椭圆曲线：PRIME192V3 */
    CRYPTO_EC_GROUPID_X9_62_PRIME239V1, /* < 椭圆曲线：PRIME239V1 */
    CRYPTO_EC_GROUPID_X9_62_PRIME239V2, /* < 椭圆曲线：PRIME239V2 */
    CRYPTO_EC_GROUPID_X9_62_PRIME239V3, /* < 椭圆曲线：PRIME239V3 */

    CRYPTO_EC_GROUPID_BRAINPOOL_P256R1, /* < 椭圆曲线：BRAINPOOLP256R1 */
    CRYPTO_EC_GROUPID_BRAINPOOL_P384R1, /* < 椭圆曲线：BRAINPOOLP384R1 */

    CRYPTO_EC_GROUPID_SM2_PRIME256, /* < 椭圆曲线：SM2 PRIME256 */

    CRYPTO_EC_GROUPID_SEC_P192R1, /* < 椭圆曲线：SECP192R1 */
    CRYPTO_EC_GROUPID_SEC_P256R1, /* < 椭圆曲线：SECP256R1 */
    CRYPTO_EC_GROUPID_SEC_P384R1, /* < 椭圆曲线：SECP384R1 */
    CRYPTO_EC_GROUPID_SEC_P521R1, /* < 椭圆曲线：SECP521R1 */

    CRYPTO_EC_GROUPID_X25519, /* < 椭圆曲线：X25519 */
    CRYPTO_EC_GROUPID_END,    /* < 最大枚举值 */
} CryptoEcGroupId;

/* 根据rfc4492定义，有些ECC曲线是相同的，为了给产品提供统一的界面，将枚举的值修改为相同。 */
#define CRYPTO_EC_GROUPID_X9_62_PRIME192V1 CRYPTO_EC_GROUPID_SEC_P192R1 /* < 椭圆曲线：PRIME192V1 */
#define CRYPTO_EC_GROUPID_X9_62_PRIME256V1 CRYPTO_EC_GROUPID_SEC_P256R1 /* < 椭圆曲线：PRIME256V1 */

#define CRYPTO_EC_GROUPID_NIST_P256 CRYPTO_EC_GROUPID_SEC_P256R1        /* < 椭圆曲线：NIST P256 */
#define CRYPTO_EC_GROUPID_NIST_P384 CRYPTO_EC_GROUPID_SEC_P384R1        /* < 椭圆曲线：NIST P384 */
#define CRYPTO_EC_GROUPID_NIST_P521 CRYPTO_EC_GROUPID_SEC_P521R1        /* < 椭圆曲线：NIST P521 */

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_CRYPTO_ECC_API_H */
