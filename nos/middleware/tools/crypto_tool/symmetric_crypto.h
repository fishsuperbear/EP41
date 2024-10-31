/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: dbg
 * Created on: May 8, 2023
 */

#ifndef SYMMETRIC_CRYPTO_H
#define SYMMETRIC_CRYPTO_H

#include <memory>
#include <vector>
#include "log/include/logging.h"
#include "nvpkcs11.h"
#include "nvpkcs11_public_defs.h"


namespace hozon {
namespace netaos {
namespace crypto {

using namespace hozon::netaos::log;

class SymmetricCrypto {
public:
    SymmetricCrypto();
    virtual ~SymmetricCrypto();

    uint32_t Init();
    void DeInit();

    void Help();
    bool encrypto(std::vector<std::string>& files);
    uint32_t encrypto(const std::string filename);
    bool decrypto(std::vector<std::string>& files);
    bool decrypto(std::string filename);
private:
    std::shared_ptr<CK_OBJECT_HANDLE>  GenerateSymmetricKey();
    bool DeriveSymmetricKey();
    uint32_t log_in();
    CK_SESSION_HANDLE session_ = 0;
    std::shared_ptr<CK_OBJECT_HANDLE> phkey_;
    CK_OBJECT_HANDLE skey_ = CK_INVALID_HANDLE;
    CK_OBJECT_HANDLE drived_key_ = CK_INVALID_HANDLE;
    CK_RV find_object_derive_key(
		CK_OBJECT_HANDLE_PTR derived_key_handle_ptr,
		std::string input_base_string,
		std::string key_id_input_string,
		std::string key_derivation_label_string,
		std::string key_derivation_context_string);
    CK_RV find_base_key_object(CK_SESSION_HANDLE hSession, CK_OBJECT_HANDLE_PTR base_key_obj_handle_ptr, 
    CK_ATTRIBUTE_PTR key_obj_template_ptr, CK_ULONG num_entries, CK_ULONG_PTR obj_count_ptr);
};

}}}

#endif