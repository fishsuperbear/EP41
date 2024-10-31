/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: crypto server loger
 */

#pragma once

#include "common/inner_types.h"
#include "cryp/cryobj/crypto_primitive_id.h"
#include "cryp/cryobj/crypto_object.h"
#include "idl/generated/crypto.h"
#include "idl/generated/cryptoPubSubTypes.h"
#include "keys/key_slot_prototype_props.h"
#include "keys/key_slot_content_props.h"

namespace hozon {
namespace netaos {
namespace crypto {

// struct PrimitiveIdInfo;
// struct CmCryptoObjectInfo;
// struct RestrictedUseInfo;
// struct KeyInfo;
// struct CryptoKeyRef;
// struct CipherCtxRef;

// class CmPrimitiveIdInfo;
// class CryptoObjectInfo;
// class CmRestrictedUseInfo;
// class CmKeyInfo;
// class CmCryptoKeyRef;
// class CmCipherCtxRef;

class TypeConverter {
public:

static void CmStructToInnerType(CmPrimitiveIdInfo& cm_struct, cryp::CryptoPrimitiveId::PrimitiveIdInfo& inner_type);
static void CmStructToInnerType(CmCryptoObjectInfo& cm_struct, cryp::CryptoObject::CryptoObjectInfo& inner_type);
// static void CmStructToInnerType(CmRestrictedUseInfo& cm_struct, RestrictedUseInfo& inner_type);
// static void CmStructToInnerType(CmKeyInfo& cm_struct, KeyInfo& inner_type);
static void CmStructToInnerType(CmCryptoKeyRef& cm_struct, CryptoKeyRef& inner_type);
static void CmStructToInnerType(CmCryptoServiceInfo& cm_struct, cryp::CryptoService::CryptoServiceInfo& inner_type);
static void CmStructToInnerType(CmCipherCtxRef& cm_struct, CipherCtxRef& inner_type);

static void InnerTypeToCmStruct(cryp::CryptoPrimitiveId::PrimitiveIdInfo& inner_type, CmPrimitiveIdInfo& cm_struct);
static void InnerTypeToCmStruct(cryp::CryptoObject::CryptoObjectInfo& inner_type, CmCryptoObjectInfo& cm_struct);
// static void InnerTypeToCmStruct(RestrictedUseInfo& inner_type, CmRestrictedUseInfo& cm_struct);
// static void InnerTypeToCmStruct(KeyInfo& inner_type, CmKeyInfo& cm_struct);
static void InnerTypeToCmStruct(CryptoKeyRef& inner_type, CmCryptoKeyRef& cm_struct);
static void InnerTypeToCmStruct(cryp::CryptoService::CryptoServiceInfo& inner_type, CmCryptoServiceInfo& cm_struct);
static void InnerTypeToCmStruct(CipherCtxRef& inner_type, CmCipherCtxRef& cm_struct);

};

}
}
}