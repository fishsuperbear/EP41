/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:加解密功能对外接口
 * Create: 2020/05/08
 * History:
 */
#ifndef ADAPTOR_CRYPTO_REG_API_H
#define ADAPTOR_CRYPTO_REG_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "keys/keys_api.h"
#include "keys/pkey_api.h"
#include "crypto/crypto_hash_api.h"
#include "crypto/crypto_cipher_api.h"
#include "crypto/crypto_asycipher_api.h"
#include "crypto/crypto_dh_api.h"
#include "crypto/crypto_ecc_api.h"
#include "crypto/crypto_mac_api.h"
#include "crypto/crypto_kdf_api.h"
#include "crypto/crypto_sign_api.h"
#include "crypto/crypto_bke_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup adaptor
 * @brief 设置AEAD算法附加验证数据的钩子。
 *
 * @param handle [IN] Crypto Cipher handle
 * @param auth [IN] AEAD 算法AAD数据。
 * @param authLen [IN] AEAD 算法AAD数据长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoCipherSetAuthFunc)(CryptoCipherHandle handle, const uint8_t *auth, size_t authLen);

/**
 * @ingroup adaptor
 * @brief 加密套件设置 Nonce 的钩子。
 *
 * @param handle [IN] Crypto Cipher handle。
 * @param nonce [OUT] nonce 数据
 * @param nonceLen [IN] nonce 数据长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoCipherSetNonceFunc)(CryptoCipherHandle handle, const uint8_t *nonce, size_t nonceLen);

/**
 * @ingroup adaptor
 * @brief 初始化加解密功能的钩子。
 *
 * @param  handle [IN] Crypto Cipher 上下文
 * @param  alg [IN] 加密算法
 * @param  isEnc [IN] 加解密标志位
 * @param  key [IN] 密钥句柄
 * @param  nonce [IN] 加密的 nonce 参数
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoCipherInitFunc)(CryptoCipherHandle *handle, CryptoCipherAlgorithm cipher, bool isEnc,
                                        KeysKeyRoHandle key, KeysKeyRoHandle nonce);

/**
 * @ingroup adaptor
 * @brief 加密套件反初始化的钩子。
 *
 * @param handle [IN] Crypto Cipher handle。
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoCipherDeinitFunc)(CryptoCipherHandle handle);

/**
 * @ingroup adaptor
 * @brief 加密套件加密的钩子。
 *
 * @param handle [IN] Crypto Cipher 上下文
 * @param text [IN] 待加密的信息
 * @param textLen [IN] 待加密的信息长度
 * @param cipherText [OUT] 加密后的信息
 * @param pCipherTextLen [OUT] 加密后的信息长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoCipherEncryptFunc)(CryptoCipherHandle handle, const uint8_t *text, size_t textLen,
                                           uint8_t *cipherText, size_t *pCipherTextLen);

/**
 * @ingroup adaptor
 * @brief 加密套件解密的钩子。
 *
 * @param handle [IN] Crypto Cipher 上下文
 * @param cipherText [IN] 待解密的信息
 * @param cipherTextLen [IN] 待解密的信息长度
 * @param text [OUT] 解密后的信息
 * @param pTextLen [OUT] 解密后的信息长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoCipherDecryptFunc)(CryptoCipherHandle handle, const uint8_t *cipherText, size_t cipherTextLen,
                                           uint8_t *text, size_t *pTextLen);

/**
 * @ingroup adaptor
 * @brief 非对称加密套件初始化的钩子。
 *
 * @param handle [OUT] Crypto Asymmetric cipher handle。
 * @param alg [IN] 算法
 * @param isEnc [IN] 是否为加密
 * @param key [IN] key handle
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoAsycipherInitFunc)(CryptoAsycipherHandle *handle, CryptoAsycipherAlgorithm alg, bool isEnc,
                                           KeysKeyRoHandle key);

/**
 * @ingroup adaptor
 * @brief 非对称加密套件反初始化的钩子。
 *
 * @param handle [IN] Crypto Asymmetric cipher handle。
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoAsycipherDeinitFunc)(CryptoAsycipherHandle handle);

/**
 * @ingroup adaptor
 * @brief 非对称加密套件加密的钩子。
 *
 * @param handle [IN] Crypto Asymmetric cipher 上下文
 * @param text [IN] 待加密的信息
 * @param textLen [IN] 待加密的信息长度
 * @param cipherText [OUT] 加密后的信息
 * @param pCipherTextLen [OUT] 加密后的信息长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoAsycipherEncryptFunc)(CryptoAsycipherHandle handle, const uint8_t *text, size_t textLen,
                                              uint8_t *cipherText, size_t *cipherTextLen);

/**
 * @ingroup adaptor
 * @brief 非对称加密套件解密的钩子。
 *
 * @param handle [IN] Crypto Asymmetric cipher 上下文
 * @param cipherText [IN] 待解密的信息
 * @param cipherTextLen [IN] 待解密的信息长度
 * @param text [OUT] 解密后的信息
 * @param textLen [OUT] 解密后的信息长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoAsycipherDecryptFunc)(CryptoAsycipherHandle handle, const uint8_t *cipherText,
                                              size_t cipherTextLen, uint8_t *text, size_t *textLen);

/**
 * @ingroup adaptor
 * @brief 非对称加密信息函数，使用公钥实现信息的加密
 *
 * @param key [IN] 非对称密钥
 * @param text [IN] 待加密的信息
 * @param textLen [IN] 待加密的信息长度
 * @param cipherText [OUT] 加密后的信息
 * @param cipherTextLen [OUT] 加密后的信息长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoAsyEncryptFunc)(PKeyAsymmetricKeyRoHandle key, const uint8_t *text,
                                        size_t textLen, uint8_t *cipherText, size_t *cipherTextLen);

/**
 * @ingroup adaptor
 * @brief 非对称解密信息函数，使用公钥实现信息的解密
 *
 * @param key [IN] 非对称密钥
 * @param cipherText [IN] 待解密的信息
 * @param cipherTextLen [IN] 待解密的信息长度
 * @param text [OUT] 解密后的信息
 * @param textLen [OUT] 解密后的信息长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoAsyDecryptFunc)(PKeyAsymmetricKeyRoHandle key, const uint8_t *cipherText,
                                        size_t cipherTextLen, uint8_t *text, size_t *textLen);

/**
 * @ingroup adaptor
 * @brief 蝴蝶算法扩展密钥。
 *
 * @param funcType [IN] 蝴蝶算法扩展密钥计算密钥类型
 * @param derivationSuitKeys [IN] 蝴蝶算法密钥套件
 * @param i [IN] 扩展密钥i
 * @param j [IN/OUT] 扩展密钥j
 * @param deribateKey [OUT] 扩展密钥
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoBkePrivateKeyDerivation)(BKEDerivationFuncType funcType, BkeSuitKeysRoHandle derivationSuitKeys,
                                                 uint32_t i, uint32_t j, KeysKeyHandle *deribateKey);

/**
 * @ingroup adaptor
 * @brief 蝴蝶算法计算最终私钥。
 *
 * @param bij [IN] 签名扩展私钥
 * @param cij [IN] 待计算私钥数据
 * @param cijLen [IN] 待计算私钥数据长度
 * @param sij [OUT] 最终私钥
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoBkePrivateKeyComput)(KeysKeyHandle bij, uint8_t *cij, size_t cijLen, KeysKeyHandle *sij);

/**
 * @ingroup adaptor
 * @brief KDF Key Derive 的钩子。
 *
 * @param alg [IN] KDF 算法
 * @param deriveInfo [IN] 派生的密钥信息
 * @param handle [OUT] 派生结果
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoKdfKeyDeriveFunc)(CryptoKdfAlgorithm alg, CryptoKeyDeriveParamHandle deriveInfo,
                                          KeysKeyHandle *handle);

/**
 * @ingroup adaptor
 * @brief KDF Key Diversifier 的钩子。
 *
 * @param handle [IN] Crypto Cipher 上下文
 * @param childNum [IN] 发散后的密钥数量
 * @param childLen [OUT] 发散后各密钥的长度
 * @param childHandle [OUT] 发散后的密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoKdfKeyDiversifierFunc)(KeysKeyHandle handle, size_t childNum, const size_t childLen[],
                                               KeysKeyHandle childHandle[]);

/**
 * @ingroup adaptor
 * @brief DH 密钥对生成的钩子
 *
 * @param dhParam [IN] DH参数
 * @param priKeyHandle [OUT] 密钥句柄
 * @param pubKeyBuf [OUT] 生成密钥缓存
 * @param size [OUT] 生成密钥的长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoDhKeyPairGenFunc)(CryptoDhParamRoHandle dhParam, KeysKeyHandle *priKeyHandle,
                                          uint8_t *pubKeyBuf, size_t *size);

/**
 * @ingroup adaptor
 * @brief DH 共享密钥计算的钩子
 *
 * @param priKeyHandle [IN] 本端私钥密钥句柄
 * @param peerPubKeyBuf [OUT] 对端公钥
 * @param size [OUT] 对端公钥长度
 * @param pskHandle [OUT] 预主密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoDhSharedkeyCalcFunc)(KeysKeyRoHandle priKeyHandle, const uint8_t *peerPubKeyBuf, size_t size,
                                             KeysKeyHandle *shareKeyHandle);

/**
 * @ingroup adaptor
 * @brief ECDH 密钥对生成的钩子。
 *
 * @param curve [IN] 椭圆曲线
 * @param priKeyHandle [OUT] 密钥句柄
 * @param pubKeyBuf [OUT] 生成密钥缓存
 * @param size [OUT] 生成密钥的长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoEcdhKeyPairGenFunc)(CryptoEcGroupId curve, KeysKeyHandle *priKeyHandle, uint8_t *pubKeyBuf,
                                            size_t *size);

/**
 * @ingroup adaptor
 * @brief ECDH 共享密钥计算的钩子。
 *
 * @param priKeyHandle [IN] 本端私钥密钥句柄
 * @param peerPubKeyBuf [OUT] 对端公钥
 * @param size [OUT] 对端公钥长度
 * @param pskHandle [OUT] 预主密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoEcdhSharedkeyCalcFunc)(KeysKeyRoHandle priKeyHandle, const uint8_t *peerPubKeyBuf, size_t size,
                                               KeysKeyHandle *pskHandle);

/**
 * @ingroup adaptor
 * @brief ECDH 公钥长度获取的钩子。
 *
 * @param curve [IN] 椭圆曲线
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef size_t (*CryptoEcdhPubkeyLenGetFunc)(CryptoEcGroupId curve);

/**
 * @ingroup adaptor
 * @brief 获取 MAC 长度的钩子。
 *
 * @param algorithm [IN] MAC 算法
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef size_t (*CryptoMacGetLenFunc)(CryptoMacAlgorithm algorithm);

/**
 * @ingroup adaptor
 * @brief 初始化 MAC 的钩子。
 *
 * @param handle [OUT] MAC 句柄
 * @param algorithm [IN] MAC 算法
 * @param key [IN] 密钥句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoMacInitFunc)(CryptoMacHandle *handle, CryptoMacAlgorithm algorithm, KeysKeyRoHandle key);

/**
 * @ingroup adaptor
 * @brief 反初始化 MAC 的钩子。
 *
 * @param handle [IN] MAC 句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoMacDeinitFunc)(CryptoMacHandle handle);

/**
 * @ingroup adaptor
 * @brief 更新 MAC 的钩子。
 *
 * @param handle [IN] MAC 句柄
 * @param data [IN] 更新的数据
 * @param len [IN] 更新的数据长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoMacUpdateFunc)(CryptoMacHandle handle, const uint8_t *data, size_t len);

/**
 * @ingroup adaptor
 * @brief MAC final 的钩子。
 *
 * @param handle [IN] MAC 句柄
 * @param digest [OUT] mac计算后的结果
 * @param digestLen [OUT] mac计算后的结果
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoMacFinalFunc)(CryptoMacHandle handle, uint8_t *digest, size_t *digestLen);

/**
 * @ingroup adaptor
 * @brief 随机字节的钩子。
 *
 * @param buf [OUT] 随机字节 buffer
 * @param num [IN] 随机字节 buffer 长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoRndRandbytesFunc)(uint8_t *buf, size_t num);

/**
 * @ingroup adaptor
 * @brief 范围随机字节的钩子。
 *
 * @param randomRange [IN] 随机数范围
 * @param resultNumber [OUT] 生成的随机数
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoRndRangeNumberFunc)(uint32_t randomRange, uint32_t *resultNumber);

/**
 * @ingroup adaptor
 * @brief 获取 SHA 长度的钩子。
 *
 * @param algorithm [IN] 哈希算法
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef size_t (*CryptoShaGetLenFunc)(CryptoHashAlgorithm algorithm);

/**
 * @ingroup adaptor
 * @brief SHA 初始化的钩子。
 *
 * @param handle [OUT] 哈希句柄
 * @param algorithm [IN] 哈希算法
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoShaInitFunc)(CryptoHashHandle *handle, CryptoHashAlgorithm algorithm);

/**
 * @ingroup adaptor
 * @brief SHA 反初始化的钩子。
 *
 * @param handle [IN] 哈希句柄
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoShaDeinitFunc)(CryptoHashHandle handle);

/**
 * @ingroup adaptor
 * @brief   SHA 更新的钩子。
 *
 * @param   handle [IN] 哈希句柄
 * @param   data [IN] 待计算哈希的内容
 * @param   len [IN] 待计算哈希的内容的长度
 *
 * @return  参见回调函数 shaUpdateCb() 的返回值
 */
typedef int32_t (*CryptoShaUpdateFunc)(CryptoHashHandle handle, const uint8_t *data, size_t len);

/**
 * @ingroup adaptor
 * @brief SHA 获取 Final 的钩子。
 *
 * @param handle [IN] 哈希句柄
 * @param digest [OUT] 哈希值
 * @param digestLen [OUT] 哈希值长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoShaFinalFunc)(CryptoHashHandle handle, uint8_t *digest, size_t *digestLen);

/**
 * @ingroup adaptor
 * @brief SHA 计算的钩子。
 *
 * @param algorithm [IN] 哈希算法
 * @param data [IN] 待计算哈希的内容
 * @param len [IN] 待计算哈希的内容的长度
 * @param digest [OUT] 计算的哈希值结果
 * @param digestLen [OUT] 计算的哈希值结果长度
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoShaCalcFunc)(CryptoHashAlgorithm algorithm, const uint8_t *data, size_t len, uint8_t *digest,
                                     size_t *digestLen);

/**
 * @ingroup adaptor
 * @brief 签名的钩子。
 *
 * @param sigAlg [IN] 签名算法
 * @param key [IN] 签名密钥
 * @param signParam [IN/OUT] @see CryptoSignParam
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoSignSignFunc)(CryptoSignAlgorithm sigAlg, PKeyAsymmetricKeyRoHandle key,
                                      CryptoSignParamHandle signInfo);

/**
 * @ingroup adaptor
 * @brief 验签的钩子。
 *
 * @param sigAlg [IN] 签名算法
 * @param key [IN] 签名密钥
 * @param signInfo [IN/OUT] @see CryptoSignParam
 *
 * @retval int32_t, 由回调函数返回的错误码
 */
typedef int32_t (*CryptoSignVerifyFunc)(CryptoSignAlgorithm sigAlg, PKeyAsymmetricKeyRoHandle key,
                                        CryptoSignParamRoHandle signInfo);

/**
 * @ingroup adaptor
 * CryptoAdaptHandleFunc 结构体，密码适配层功能钩子函数集
 */
typedef struct {
    /* symmetric encryption and decryption */
    CryptoCipherSetAuthFunc cipherSetAuthCb;   /* < 加密套件设置 Auth 的钩子 */
    CryptoCipherSetNonceFunc cipherSetNonceCb; /* < 加密套件设置 Nonce 的钩子 */
    CryptoCipherInitFunc cipherInitCb;         /* < 加密套件初始化的钩子 */
    CryptoCipherDeinitFunc cipherDeInitCb;     /* < 加密套件反初始化的钩子 */
    CryptoCipherEncryptFunc cipherEncryptCb;   /* < 加密套件加密的钩子 */
    CryptoCipherDecryptFunc cipherDecryptCb;   /* < 加密套件解密的钩子 */

    /* asymmetric decryption */
    CryptoAsycipherInitFunc asycipherInitCb;       /* < 非对称加密套件初始化的钩子 */
    CryptoAsycipherDeinitFunc asycipherDeInitCb;   /* < 非对称加密套件反初始化的钩子 */
    CryptoAsycipherEncryptFunc asycipherEncryptCb; /* < 非对称加密套件加密的钩子 */
    CryptoAsycipherDecryptFunc asycipherDecryptCb; /* < 非对称加密套件解密的钩子 */
    CryptoAsyEncryptFunc asyEncryptCb;             /* < 非对称加密的钩子 */
    CryptoAsyDecryptFunc asyDecryptCb;             /* < 非对称解密的钩子 */

    /* bke */
    CryptoBkePrivateKeyDerivation bkePrivateKeyDerivationCb; /* < 蝴蝶算法扩展密钥的钩子 */
    CryptoBkePrivateKeyComput bkePrivateKeyComputCb;         /* < 蝴蝶算法计算最终私钥的钩子 */

    /* key derive */
    CryptoKdfKeyDeriveFunc kdfKeyDeriveCb;           /* < KDF Key Derive 的钩子 */
    CryptoKdfKeyDiversifierFunc kdfKeyDiversifierCb; /* < KDF Key Diversifier 的钩子 */

    /* dh */
    CryptoDhKeyPairGenFunc dhKeyPairGenCb;           /* < DH 密钥对生成的钩子 */
    CryptoDhSharedkeyCalcFunc dhSharedKeyCalcCb;     /* < DH 共享密钥计算的钩子 */
    /* ecc */
    CryptoEcdhKeyPairGenFunc ecdhKeyPairGenCb;       /* < ECDH 密钥对生成的钩子 */
    CryptoEcdhSharedkeyCalcFunc ecdhSharedKeyCalcCb; /* < ECDH 共享密钥计算的钩子 */
    CryptoEcdhPubkeyLenGetFunc ecdhPubKeyLenGetCb;   /* < ECDH 公钥长度获取的钩子 */
    /* mac calc */
    CryptoMacGetLenFunc macGetLenCb; /* < 获取 MAC 长度的钩子 */
    CryptoMacInitFunc macInitCb;     /* < 初始化 MAC 的钩子 */
    CryptoMacDeinitFunc macDeInitCb; /* < 反初始化 MAC 的钩子 */
    CryptoMacUpdateFunc macUpdateCb; /* < 更新 MAC 的钩子 */
    CryptoMacFinalFunc macFinalCb;   /* < MAC final 的钩子 */
    /* random */
    CryptoRndRandbytesFunc rndRandBytesCb;     /* < 随机字节的钩子 */
    CryptoRndRangeNumberFunc rndRangeNumberCb; /* < 范围随机数的钩子 */
    /* hash calc */
    CryptoShaGetLenFunc shaGetLenCb; /* < 获取 SHA 长度的钩子 */
    CryptoShaInitFunc shaInitCb;     /* < SHA 初始化的钩子 */
    CryptoShaDeinitFunc shaDeInitCb; /* < SHA 反初始化的钩子 */
    CryptoShaUpdateFunc shaUpdateCb; /* < SHA 更新的钩子 */
    CryptoShaFinalFunc shaFinalCb;   /* < SHA 获取 Final 的钩子 */
    CryptoShaCalcFunc shaCalcCb;     /* < SHA 计算的钩子 */
    /* sign */
    CryptoSignSignFunc signSignCb;     /* < 签名的钩子 */
    CryptoSignVerifyFunc signVerifyCb; /* < 验签的钩子 */
} CryptoAdaptHandleFunc;

#ifdef __cplusplus
}
#endif

#endif /* ADAPTOR_CRYPTO_REG_API_H */
