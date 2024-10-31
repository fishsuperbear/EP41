#ifndef ARA_CRYPTO_INNER_TYPES_H_
#define ARA_CRYPTO_INNER_TYPES_H_

#include <string>

#include "common/base_id_types.h"
#include "cryp/crypto_service.h"
#include "cryp/cryobj/crypto_object.h"
#include "cryp/cryobj/restricted_use_object.h"

namespace hozon {
namespace netaos {
namespace crypto {

// struct PrimitiveIdInfo {
// 	uint64_t     alg_id;
// };

// // Key property定义
// struct CryptoObjectInfo
// {
// 	uint32_t object_type;
// 	uint32_t dependency_type;
//     std::string object_uid;
//     std::string dependency_uid;
// 	uint64_t payload_size;
// 	bool is_session;
// 	bool is_exportable;
// };

// struct RestrictedUseInfo {
// 	uint32_t allowed_usage;
// };

// struct KeyInfo {
//     uint64_t key_type;
// };

enum CipherContextType : uint32_t
{
	kCipherContextType_UnDefined = 0,
	kCipherContextType_EncryptorPublic,
	kCipherContextType_DecryptorPrivate,
	kCipherContextType_HashFunction,
	kCipherContextType_SignerPrivate,
	kCipherContextType_SymmetricBlockCipher,
	kCipherContextType_VerifierPublic,
	kCipherContextType_SignPrehash
};

enum KeyType : uint32_t
{
  kKeyType_None = 0,
  kKeyType_SymmetricKey,
	kKeyType_PrivateKey,
  kKeyType_PublicKey,
};


// key(引用)定义
struct CryptoKeyRef
{
    int32_t alg_id;       // alg_id可以确定算法类型
    // int32_t key_type;     // key_type. 0: none, 1: symmetric key, 2: public key, 3: private key.
    uint64_t ref;         // 对server端key实体的引用。引用内容由server解释，client端不感知
	cryp::CryptoPrimitiveId::PrimitiveIdInfo primitive_id_info;
    cryp::CryptoObject::CryptoObjectInfo crypto_object_info;
	AllowedUsageFlags allowed_usage;
	// KeyInfo key_info;

	CryptoKeyRef()
	: alg_id(kAlgIdUndefined)
	, ref(0)
	, primitive_id_info()
	, crypto_object_info()
	, allowed_usage(kAllowPrototypedOnly) {

	}

	CryptoKeyRef(const CryptoKeyRef& other)
	: alg_id(other.alg_id)
	, ref(other.ref)
	, primitive_id_info(other.primitive_id_info)
	, crypto_object_info(other.crypto_object_info)
	, allowed_usage(other.allowed_usage) {

	}

	CryptoKeyRef& operator= (const CryptoKeyRef& other) {
		alg_id = other.alg_id;
		ref = other.ref;
		primitive_id_info = other.primitive_id_info;
		crypto_object_info = other.crypto_object_info;
		allowed_usage = other.allowed_usage;
		return *this;
	}
};

// 密码学操作context引用定义
struct CipherCtxRef
{
	int32_t alg_id;       // alg_id可以确定算法类型
    int32_t ctx_type;
	uint64_t ref;         // 对server端context实体的引用。引用内容由server解释，client端不感知
    int32_t transform;
	bool is_initialized;

	cryp::CryptoService::CryptoServiceInfo crypto_service_info;

	CipherCtxRef()
	: alg_id(kAlgIdUndefined)
	, ctx_type(kCipherContextType_UnDefined)
	, ref(0)
	, transform(0)
	, is_initialized(false)
	, crypto_service_info() {

	}

	CipherCtxRef(const CipherCtxRef& other)
	: alg_id(other.alg_id)
	, ctx_type(other.ctx_type)
	, ref(other.ref)
	, transform(other.transform)
	, is_initialized(other.is_initialized)
	, crypto_service_info(other.crypto_service_info) {

	}

	CipherCtxRef& operator= (const CipherCtxRef& other) {
		alg_id = other.alg_id;
		ctx_type = other.ctx_type;
		ref = other.ref;
		transform = other.transform;
		is_initialized = other.is_initialized;
		crypto_service_info = other.crypto_service_info;
		return *this;
	}
};

// Key slot定义
struct CryptoSlotRef
{
	std::string uuid;
	uint64_t ref;
};

struct CryptoIoContainerRef
{
	uint64_t ref;
};

// cert(引用)定义
struct CryptoCertRef
{
    uint64_t ref;  
	CryptoCertRef():ref(0){}
};

// DN(引用)定义
struct X509DNRef
{
    uint64_t ref;  
	X509DNRef():ref(0){}
};

}
}
}
#endif