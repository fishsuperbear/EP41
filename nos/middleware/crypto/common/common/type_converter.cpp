/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: crypto server loger
 */

#include "common/type_converter.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {

void TypeConverter::CmStructToInnerType(CmPrimitiveIdInfo& cm_struct, cryp::CryptoPrimitiveId::PrimitiveIdInfo& inner_type) {
    inner_type.alg_id = cm_struct.alg_id();
}

void TypeConverter::CmStructToInnerType(CmCryptoObjectInfo& cm_struct, cryp::CryptoObject::CryptoObjectInfo& inner_type) {
	inner_type.objectUid.mCOType = static_cast<CryptoObjectType>(cm_struct.object_type());
	inner_type.dependencyUid.mCOType = static_cast<CryptoObjectType>(cm_struct.dependency_type());
    CRYP_INFO<<"cm_struct.object_uid:"<<cm_struct.object_uid();
    auto uuid_res = FromString(cm_struct.object_uid());
    auto dep_uuid_res = FromString(cm_struct.dependency_uid());
    CRYP_INFO<<"CmStructToInnerType: 1";
    if (uuid_res) {
        inner_type.objectUid.mCouid = CryptoObjectUid { uuid_res.Value(), 4 } ;
        CRYP_INFO<<"CmStructToInnerType:2";
    }else {
        CRYP_ERROR<<"CmStructToInnerType:3 .Make object uuid failed.";
        // CRYP_ERROR << "Make object uuid failed. " << CRYP_ERROR_MESSAGE(uuid_res.Error().Value());
    }

    if (dep_uuid_res) {
        inner_type.dependencyUid.mCouid = CryptoObjectUid { dep_uuid_res.Value(), 4 };
        CRYP_INFO<<"CmStructToInnerType:4";

    }else if (!cm_struct.dependency_uid().empty()) {
        CRYP_ERROR<<"CmStructToInnerType:5.Make dependency uuid failed.";
        // CRYP_INFO << "Make dependency uuid failed. " << CRYP_ERROR_MESSAGE(dep_uuid_res.Error().Value());
    }else {
        CRYP_INFO << "CmStructToInnerType:6";
    }
    inner_type.payloadSize = cm_struct.payload_size();
    inner_type.isExportable = cm_struct.is_exportable();
	inner_type.isSession = cm_struct.is_session();
    CRYP_INFO<<"CmStructToInnerType:7";

}

// void TypeConverter::CmStructToInnerType(CmRestrictedUseInfo& cm_struct, RestrictedUseInfo& inner_type) {
//     inner_type.allowed_usage = cm_struct.allowed_usage();
// }

// void TypeConverter::CmStructToInnerType(CmKeyInfo& cm_struct, KeyInfo& inner_type) {
//     inner_type.key_type = cm_struct.key_type();
// }

void TypeConverter::CmStructToInnerType(CmCryptoKeyRef& cm_struct, CryptoKeyRef& inner_type) {
    inner_type.alg_id = cm_struct.alg_id();
    inner_type.ref = cm_struct.ref();
    // cryp::CryptoPrimitiveId::PrimitiveIdInfo  &pri_id_info = inner_type.primitive_id_info;
    // cryp::CryptoObject::CryptoObjectInfo  &cry_obj_info = inner_type.crypto_object_info;
    CmStructToInnerType(cm_struct.primitive_id_info(), inner_type.primitive_id_info);
    CmStructToInnerType(cm_struct.crypto_object_info(), inner_type.crypto_object_info);
    //    CmStructToInnerType(cm_struct.primitive_id_info(), pri_id_info);
    // CmStructToInnerType(cm_struct.crypto_object_info(), cry_obj_info);
    inner_type.allowed_usage = cm_struct.allowed_usage();
}

void TypeConverter::CmStructToInnerType(CmCryptoServiceInfo& cm_struct, cryp::CryptoService::CryptoServiceInfo& inner_type) {
    inner_type.block_size = cm_struct.block_size();
    inner_type.max_input_size = cm_struct.max_input_size();
    inner_type.max_output_size = cm_struct.max_output_size();
}

void TypeConverter::CmStructToInnerType(CmCipherCtxRef& cm_struct, CipherCtxRef& inner_type) {
	inner_type.alg_id = cm_struct.alg_id();
    inner_type.ctx_type = cm_struct.ctx_type();
	inner_type.ref = cm_struct.ref();
    inner_type.transform = cm_struct.transform();
    inner_type.is_initialized = cm_struct.is_initialized();
    CmStructToInnerType(cm_struct.crypto_service_info(), inner_type.crypto_service_info);
}

void TypeConverter::InnerTypeToCmStruct(cryp::CryptoPrimitiveId::PrimitiveIdInfo& inner_type, CmPrimitiveIdInfo& cm_struct) {
    cm_struct.alg_id(inner_type.alg_id);
}

void TypeConverter::InnerTypeToCmStruct(cryp::CryptoObject::CryptoObjectInfo& inner_type, CmCryptoObjectInfo& cm_struct) {
	cm_struct.object_type(static_cast<uint32_t>(inner_type.objectUid.mCOType));
	cm_struct.dependency_type(static_cast<uint32_t>(inner_type.dependencyUid.mCOType));
    cm_struct.object_uid(inner_type.objectUid.mCouid.mGeneratorUid.ToUuidStr());
    cm_struct.dependency_uid(inner_type.dependencyUid.mCouid.mGeneratorUid.ToUuidStr());
    cm_struct.payload_size(inner_type.payloadSize);
    cm_struct.is_exportable(inner_type.isExportable);
	cm_struct.is_session(inner_type.isSession);
}

// void TypeConverter::InnerTypeToCmStruct(RestrictedUseInfo& inner_type, CmRestrictedUseInfo& cm_struct) {
//     cm_struct.allowed_usage(inner_type.allowed_usage);
// }

// void TypeConverter::InnerTypeToCmStruct(KeyInfo& inner_type, CmKeyInfo& cm_struct) {
//     cm_struct.key_type(inner_type.key_type);
// }

void TypeConverter::InnerTypeToCmStruct(CryptoKeyRef& inner_type, CmCryptoKeyRef& cm_struct) {
    cm_struct.alg_id(inner_type.alg_id);
    cm_struct.ref(inner_type.ref);
    cm_struct.primitive_id_info().alg_id(inner_type.primitive_id_info.alg_id);
    InnerTypeToCmStruct(inner_type.primitive_id_info, cm_struct.primitive_id_info());
    InnerTypeToCmStruct(inner_type.crypto_object_info, cm_struct.crypto_object_info());
    cm_struct.allowed_usage(inner_type.allowed_usage);
}

void TypeConverter::InnerTypeToCmStruct(cryp::CryptoService::CryptoServiceInfo& inner_type, CmCryptoServiceInfo& cm_struct) {
	cm_struct.block_size(inner_type.block_size);
    cm_struct.max_input_size(inner_type.max_input_size);
    cm_struct.max_output_size(inner_type.max_output_size);
}

void TypeConverter::InnerTypeToCmStruct(CipherCtxRef& inner_type, CmCipherCtxRef& cm_struct) {
	cm_struct.alg_id(inner_type.alg_id);
    cm_struct.ctx_type(inner_type.ctx_type);
	cm_struct.ref(inner_type.ref);
    cm_struct.transform(inner_type.transform);
    cm_struct.is_initialized(inner_type.is_initialized);
    InnerTypeToCmStruct(inner_type.crypto_service_info, cm_struct.crypto_service_info());
}

}
}
}