
#include "symmetric_crypto.h"
#include <iostream>
#include <fstream>
#include "crypto_tool_log.h"


namespace hozon {
namespace netaos {
namespace crypto {

#define SPACE_CHAR 0x20
#define REFERENCE_PKCS11_CODE_DEBUG
#define CK_SP800_108_COUNTER_FORMAT_WIDTH_IN_BITS 32UL
#define CK_SP800_REQUIRED_LENGTH_FORMAT_WIDTH 32U
#define CK_SP800_MAX_LABEL_SIZE 32U
#define CK_SP800_MAX_CONTEXT_SIZE 32U

#define REFERENCE_UNWRAP_AAD_SIZE 128U
#define REFERENCE_UNWRAP_IV_SIZE 12U
#define REFERENCE_UNWRAP_KEYTAG_SIZE 64U

/** Convert character string to byte array */
static CK_RV string_to_pkcs11_byte_array(char * input_string, CK_ULONG input_string_size, CK_BYTE_PTR output_string, CK_ULONG output_buf_size)
{
	CK_RV rv = CKR_OK;

	if(input_string_size > output_buf_size){
		std::cout<< "Error: Input string size is larger than output buffer size." <<std::endl;
		rv = CKR_DATA_LEN_RANGE;
	}
	else
	{
		for(CK_ULONG i = (CK_ULONG)0; i < output_buf_size; i++)
		{
			if(i < input_string_size)
			{
				output_string[i] = (CK_BYTE)input_string[i];
			}
			else
			{
				output_string[i] = SPACE_CHAR;
			}
		}
	}

	return rv;
}

SymmetricCrypto::SymmetricCrypto(){

	
}

SymmetricCrypto::~SymmetricCrypto(){

	C_Logout(session_);
	C_CloseSession(session_);
	C_Finalize(NULL);

}

void SymmetricCrypto::DeInit(){

}

uint32_t SymmetricCrypto::Init(){
	std::cout<< "SymmetricCrypto Init."<<std::endl;
    CK_RV rv = CKR_OK;

	CK_BBOOL token_present = (CK_BBOOL)CK_TRUE;
	CK_BBOOL token_found = (CK_BBOOL)CK_FALSE;
	CK_SLOT_ID slot_ids[NVPKCS11_TOKEN_COUNT];
	CK_ULONG slot_count = NVPKCS11_TOKEN_COUNT;
	CK_SLOT_ID ccplex_dynamic_slot_id = 0U;
	CK_ULONG i;

    CK_C_INITIALIZE_ARGS init_args =
	{
		.CreateMutex = NULL,
		.DestroyMutex = NULL,
		.LockMutex = NULL,
		.flags = (CK_FLAGS)CKF_OS_LOCKING_OK,
		.pReserved = NULL
	};

	/** Initialise Cryptoki library */
	rv = C_Initialize(&init_args);
	if(rv != CKR_OK){
		std::cout<<"C_Initialize failed."<<std::endl;
		return CKR_GENERAL_ERROR;
	}else{
		std::cout<<"C_Initialize success."<<std::endl;

	}

	rv = C_GetSlotList(token_present, slot_ids, &slot_count);
	if(rv != CKR_OK){
		std::cout<<"C_GetSlotList failed."<<std::endl;
		return CKR_GENERAL_ERROR;
	}

	for (i = 0; i < slot_count; i++)
	{
		CK_TOKEN_INFO token_info;
		rv = C_GetTokenInfo(slot_ids[i], &token_info);
		if (rv == CKR_OK)
		{
			if (strncmp((char *)token_info.model, NVPKCS11_CCPLEX_DYNAMIC_2_MODEL_NAME, sizeof(token_info.model)) == 0)
			{
				ccplex_dynamic_slot_id = slot_ids[i];
				token_found = (CK_BBOOL)CK_TRUE;
				break;
			}
		}
	}

	if(token_found == CK_FALSE){
		std::cout<<"Could not find requested token,error,exit."<<std::endl;
		rv = CKR_TOKEN_NOT_PRESENT;
		return CKR_GENERAL_ERROR;
	}else{
		std::cout<<" find requested token NVPKCS11_CCPLEX_DYNAMIC_2_MODEL_NAME success."<<std::endl;
	}

	CK_STATE session_state = CKS_RO_PUBLIC_SESSION; /* Read only session */
	rv = C_OpenSession(ccplex_dynamic_slot_id,(CK_FLAGS)(CKF_SERIAL_SESSION|session_state),NULL,NULL,&session_);
	if(rv != CKR_OK){
		std::cout<<"C_OpenSession failed."<<std::endl;
		return CKR_GENERAL_ERROR;
	}

	rv = log_in();
	if(rv != CKR_OK){
		std::cout<<"log_in failed."<<std::endl;
	}
	GenerateSymmetricKey();

    return rv;
}

void SymmetricCrypto::Help(){
	std::cout<<R"deli(
    usage:  crypto-tool
            crypto-tool -h                       Show help info.
            crypto-tool -en  filename            encrypt a file.
            crypto-tool -de  filename            decrypt a file.
)deli"<<std::endl;
}

bool SymmetricCrypto::encrypto(std::vector<std::string>& files){

	return true;

}

uint32_t SymmetricCrypto::encrypto(const std::string filename){
	CK_RV rv = CKR_OK;
    CK_BYTE encryptedData[4096] = {0x00};
    CK_ULONG encryptedDataLen = 4096;

    CK_BYTE iv[NVPKCS11_AES_CBC_IV_LEN] = {0x00};
    CK_BYTE ivLen = NVPKCS11_AES_CBC_IV_LEN;

	CK_MECHANISM mech = {
		// .mechanism = CKM_AES_CBC_PAD,
		.mechanism = CKM_AES_CBC,
		.pParameter = NULL,
		.ulParameterLen = 0,
	};

	std::cout<< "drived_key_ addr:"<<&drived_key_<<std::endl;
    rv = C_EncryptInit(session_, &mech, drived_key_);
	if(CKR_OK!=rv){
		std::cout<< "C_EncryptInit failed. ret:"<<rv <<std::endl;
		return rv;
	}else{
		std::cout<< "C_EncryptInit success." <<std::endl;
	}

    std::vector<std::uint8_t> pay;
    std::ios_base::openmode mode = std::ios::in;
    mode |= std::ios::binary;
    std::fstream fs(filename,mode);
    if(!fs.is_open()){
        std::cout<<"fs open failed."<<std::endl;
        return CKR_DATA_INVALID;
    }else{
        fs.seekg(0, std::ios::end);
        uint64_t length = fs.tellg();
        fs.seekg(0);

        pay.resize(length);
        fs.read(reinterpret_cast<char*>(pay.data()),length);
        fs.close();
    }

	if(!phkey_){
		std::cout<< "phkey_ is null ,error."<<std::endl;
		return CKR_KEY_HANDLE_INVALID;
	}

	std::cout<< "need to encrypto data:" <<std::endl;

	for(uint32_t i=0;i<pay.size();i++){
		std::cout<<pay[i]<<" ";
	}
	std::cout<<std::endl;
	std::cout << "need to encryt data size:" << pay.size()<<std::endl;

	rv = C_Encrypt(session_, static_cast<CK_BYTE_PTR>(pay.data()),pay.size(),encryptedData,&encryptedDataLen);
	if(CKR_OK != rv){
		std::cout<< "C_Encrypt failed. ret:"<<rv <<std::endl;
		return rv;
	}else{
		std::cout<< "C_Encrypt success. encryptedDataLen:"<<encryptedDataLen <<std::endl;
	}

	rv = C_NVIDIA_EncryptGetIV(session_, iv, &ivLen);
	if(CKR_OK != rv){
		std::cout<< "C_NVIDIA_EncryptGetIV failed. ret:"<<rv <<std::endl;
		return rv;
	}else{
		std::cout<<"ivLen:"<<static_cast<uint32_t>(ivLen)<<std::endl;
		std::cout<< "iv:";
		for(uint32_t i=0;i<ivLen;i++){
			std::cout<<std::hex<<static_cast<uint32_t>(iv[i])<<" ";
		}
		std::cout<<std::endl;
	}

	std::string en_file = filename + ".encrypted";
	std::string iv_file = filename + "_iv";
	std::ofstream outfile_en(en_file, std::ios::out);
    if (!outfile_en.is_open()) {
        std::cout << ":Cannot open file: " << en_file;
        return -1;
    }

	// outfile_en.write(encryptedData, encryptedDataLen);
	outfile_en<<encryptedData<<std::endl;
	outfile_en.close();

	std::ofstream outfile_iv(iv_file, std::ios::out);
    if (!outfile_iv.is_open()) {
        std::cout << ":Cannot open file: " << iv_file;
        return -1;
    }

	// outfile_iv.write(iv, ivLen);
	outfile_iv<<iv<<std::endl;
	outfile_iv.close();

	return rv;
}

bool SymmetricCrypto::decrypto(std::vector<std::string>& files){
	return true;


}

bool SymmetricCrypto::decrypto(std::string filename){
    CK_RV rv = CKR_OK;
    CK_BYTE plaintext[4096] = {0x00};
    CK_ULONG plaintextLen = 4096;

    CK_BYTE iv[NVPKCS11_AES_CBC_IV_LEN] = {0x40,0x5a,0x9e,0x5b,0x8b,0x81,0x6a,0x2f,0x2f,0x43,0xd6,0x47,0x92,0x77,0xfe,0xa8};
    CK_BYTE ivLen = NVPKCS11_AES_CBC_IV_LEN;

    std::vector<std::uint8_t> pay;
    std::ios_base::openmode mode = std::ios::in;
    mode |= std::ios::binary;
    std::fstream fs(filename,mode);
    if(!fs.is_open()){
        std::cout<<"fs open failed."<<std::endl;
        return CKR_DATA_INVALID;
    }else{
        fs.seekg(0, std::ios::end);
        uint64_t length = fs.tellg();
        fs.seekg(0);

        pay.resize(length);
        fs.read(reinterpret_cast<char*>(pay.data()),length);
        fs.close();
    }

	if(!phkey_){
		std::cout<< "phkey_ is null ,error."<<std::endl;
		return CKR_KEY_HANDLE_INVALID;
	}

	std::cout<< "need to decrypto data:" <<std::endl;

	for(uint32_t i=0;i<pay.size();i++){
		std::cout<<pay[i]<<" ";
	}
	std::cout<<std::endl;
	std::cout << "need to decryt data size:" << pay.size()<<std::endl;


	// CK_AES_CBC_ENCRYPT_DATA_PARAMS encrypt_param = {
	// 	.iv = {0x40,0x5a,0x9e,0x5b,0x8b,0x81,0x6a,0x2f,0x2f,0x43,0xd6,0x47,0x92,0x77,0xfe,0xa8},
    //     .pData = nullptr,
    //     .length = 0,
	// };

	CK_MECHANISM mech = {
		// .mechanism = CKM_AES_CBC_PAD,
		.mechanism = CKM_AES_CBC,
		.pParameter = iv,
		.ulParameterLen = NVPKCS11_AES_CBC_IV_LEN,
	};

	std::cout<< "drived_key_ addr:"<<&drived_key_<<std::endl;
    rv = C_DecryptInit(session_, &mech, drived_key_);
	if(CKR_OK!=rv){
		std::cout<< "C_DecryptInit failed. ret:"<<rv <<std::endl;
		return rv;
	}else{
		std::cout<< "C_DecryptInit success." <<std::endl;
	}

	rv = C_Decrypt(session_, static_cast<CK_BYTE_PTR>(pay.data()),pay.size() -1 ,plaintext,&plaintextLen);
	if(CKR_OK != rv){
		std::cout<< "C_Decrypt failed. ret:"<<rv <<std::endl;
		return rv;
	}else{
		std::cout<< "C_Decrypt success. plaintextLen:"<<plaintextLen <<std::endl;
	}

	std::cout<< "decrypt data:" <<std::endl;
	for(uint32_t i=0;i<plaintextLen;i++){
		std::cout<< " " <<std::hex<<plaintext[i];
	}
	std::cout<<std::endl;
	std::string de_file = filename + ".decrypted";
	std::ofstream outfile_de(de_file, std::ios::out);
    if (!outfile_de.is_open()) {
        std::cout << ":Cannot open file: " << de_file;
        return -1;
    }

	// outfile_en.write(encryptedData, encryptedDataLen);
	// outfile_de.write(static_cast<char*>(plaintext),plaintextLen);
	outfile_de.close();

	return rv;

}


std::shared_ptr<CK_OBJECT_HANDLE> SymmetricCrypto::GenerateSymmetricKey() {

	CK_RV rv = CKR_OK;
	CK_BBOOL istoken = false;
	CK_BBOOL enabled = true;
	// CK_BBOOL disabled = false;
	CK_ULONG key_len = 16; //default value
	CK_OBJECT_HANDLE symkey = 0;
    CK_OBJECT_HANDLE derived_key_handle;
	std::string input_base = "NV_OEM_KEY1";
	std::string key_id_input = "DERIVED_KEY";
	std::string key_derivation_label = "NETA_DERIVATION_LABEL";
	std::string key_derivation_context = "NETA_DERIVATION_CTX";

    rv = find_object_derive_key(&drived_key_, input_base, key_id_input,key_derivation_label, key_derivation_context);

    // std::string cka_ecid_string = "hozon_auto";
	// std::string cka_label_string = "hozon_aes";
	// CK_BYTE cka_ecid[NVPKCS11_MAX_KEY_ID_SIZE];
	// CK_BYTE cka_label[NVPKCS11_MAX_CKA_LABEL_SIZE];
	// rv = string_to_pkcs11_byte_array(const_cast<char*>(cka_ecid_string.data()), cka_ecid_string.size(), cka_ecid, sizeof(cka_ecid));
	// rv = string_to_pkcs11_byte_array(const_cast<char*>(cka_label_string.data()), cka_label_string.size(), cka_label, sizeof(cka_label));
	// CK_MECHANISM_TYPE aesMechanismList[] = {CKM_AES_CBC,CKM_AES_CBC_PAD};

	// CK_ATTRIBUTE aes_key_template[] =
	// {
	// 	// {CKA_CLASS, &pub_key_class, sizeof(pub_key_class)},
	// 	{CKA_TOKEN, &istoken, sizeof(istoken)},
	// 	{CKA_LABEL,cka_label,sizeof(cka_label)},
	// 	{CKA_ID, cka_ecid, sizeof(cka_ecid)},
	// 	{CKA_ENCRYPT, &enabled,sizeof(enabled)},
	// 	{CKA_DECRYPT, &enabled,sizeof(enabled)},
	// 	{CKA_VALUE_LEN,&key_len,sizeof(key_len)},
	// 	{CKA_ALLOWED_MECHANISMS, aesMechanismList, sizeof(aesMechanismList)}	
	// };
	// CK_ULONG attribute_count = (CK_ULONG)(sizeof(aes_key_template) / sizeof(CK_ATTRIBUTE));
	// CK_MECHANISM mech = {
	// 	.mechanism = CKM_AES_KEY_GEN,
	// 	.pParameter = nullptr,
	// 	.ulParameterLen = 0,
	// };

	// rv = C_GenerateKey(session_,&mech,aes_key_template,attribute_count,&skey_);
	// if(CKR_OK != rv){
	// 	std::cout<<"C_GenerateKey failed.rv:"<<rv<<std::endl;
	// }else{
	// 	std::cout<<"C_GenerateKey success."<<std::endl;
	// 	std::cout<< "symkey addr:"<<&skey_<<std::endl;
	// }

	// std::cout<< "key addr:"<<&skey_<<std::endl;
	phkey_ = std::make_shared<CK_OBJECT_HANDLE>(drived_key_);
	return phkey_;
}

bool SymmetricCrypto::DeriveSymmetricKey() {
	return true;


}

uint32_t SymmetricCrypto::log_in(){
	CK_RV rv = CKR_OK;
	CK_USER_TYPE user_type = CKU_USER; 
	CK_UTF8CHAR_PTR pin_ptr = NULL; 
	CK_ULONG pin_len = 0UL; 
	rv = C_Login(session_, user_type, pin_ptr, pin_len);
	return rv;
}

/**
 * Find the base key object handle so we can derive a key from it
 * @param [in] hSession Session handle
 * @param [out] base_key_obj_handle_ptr Pointer to a token-specific identifier for an object
 * @param [in] key_obj_template_ptr Pointer to a search template that specifies the attribute values to match
 * @param [in] num_entries Number of entries in the object key template
 * @param [in,out] obj_count_ptr Points to the location that receives the actual number of object handles returned
 */
CK_RV SymmetricCrypto::find_base_key_object(
		CK_SESSION_HANDLE hSession,
		CK_OBJECT_HANDLE_PTR base_key_obj_handle_ptr,
		CK_ATTRIBUTE_PTR key_obj_template_ptr,
		CK_ULONG num_entries,
		CK_ULONG_PTR obj_count_ptr)
{
	CK_RV rv = CKR_OK;
	/** Initialize a search for token and session objects that match the template */
	rv = C_FindObjectsInit(session_, key_obj_template_ptr, num_entries);
	if(rv != CKR_OK){
		CRYTOOL_ERROR << "C_FindObjectsInit failed.";
		return rv;
	}

	CK_ULONG obj_count_max = 1UL;
	/** Continue a search for token and session objects that match the template */
	rv = C_FindObjects(session_, base_key_obj_handle_ptr, obj_count_max, obj_count_ptr);
	if(rv != CKR_OK){
		CRYTOOL_ERROR << "C_FindObjects failed.";
		return rv;
	}

	if(*obj_count_ptr == 0UL){
		rv = CKR_FUNCTION_FAILED;
		CRYTOOL_ERROR << "C_FindObjects failed.No Objects found";
	}
	/** Terminates the search for token and session objects */
	CK_RV final_rv = C_FindObjectsFinal(session_);
	if(final_rv != CKR_OK){
		CRYTOOL_ERROR << "C_FindObjectsFinal Failed, returned value ="<<final_rv;
	}

	rv = final_rv;
	return rv;

}


CK_RV SymmetricCrypto::find_object_derive_key(
		CK_OBJECT_HANDLE_PTR derived_key_handle_ptr,
		std::string input_base_string,
		std::string key_id_input_string,
		std::string key_derivation_label_string,
		std::string key_derivation_context_string)
{
	CK_RV rv = CKR_OK;
	CK_OBJECT_HANDLE base_key_obj_handle = CK_INVALID_HANDLE;
	CK_OBJECT_CLASS object_class = CKO_SECRET_KEY;

	/** We derive a secret key from a token fused key.
	 * This is a 32 bytes base key called NV_OEM_KEY1.
	 * Must be space character (0x20) padded and not
	 * NULL terminated.
	 * The derive mechanism type shall be set to CKM_SP800_108_COUNTER_KDF,
	 * and the mechanism parameter to a key derivation function data structure,
	 * CK_SP800_108_KDF_PARAMS, instance.
	 * The pseudo-random function type in CK_SP800_108_KDF_PARAMS shall be
	 * set to CKM_SHA256_HMAC (.prfType = CKM_SHA256_HMAC).
	 */
	CK_BYTE base_string[NVPKCS11_MAX_KEY_ID_SIZE];
	CK_BYTE label_string[NVPKCS11_MAX_CKA_LABEL_SIZE];
	CK_BYTE context_string[NVPKCS11_MAX_CKA_LABEL_SIZE];
	if(input_base_string.empty() || (input_base_string.size() > NVPKCS11_MAX_KEY_ID_SIZE)){
		CRYTOOL_ERROR << "input_base_string is invalid.";
		return CKR_DATA_INVALID;
	}

	rv = string_to_pkcs11_byte_array(const_cast<char*>(input_base_string.data()), input_base_string.size(), base_string, sizeof(base_string));
	if(rv != CKR_OK){
		CRYTOOL_ERROR << "convert input_base_string  to base_string failed.";
	}

	// rv = string_to_pkcs11_byte_array(const_cast<char*>(key_derivation_label_string.data()), key_derivation_label_string.size(), label_string, sizeof(label_string));
	// if(rv != CKR_OK){
	// 	CRYTOOL_ERROR << "convert key_derivation_label_string  to label_string failed.";
	// }

	// rv = string_to_pkcs11_byte_array(const_cast<char*>(key_derivation_context_string.data()), key_derivation_context_string.size(), context_string, sizeof(context_string));
	// if(rv != CKR_OK){
	// 	CRYTOOL_ERROR << "convert key_derivation_label_string  to label_string failed.";
	// }
	// CRYTOOL_INFO << "label_string size:" << (CK_ULONG)sizeof(label_string) << " context_string size:" << (CK_ULONG)sizeof(context_string);

	/** key_template number of entries */
	CK_ULONG obj_count = 0UL;
	CK_ATTRIBUTE key_obj_template[] = {
			{CKA_ID, base_string, (CK_ULONG)sizeof(base_string)},
			{CKA_CLASS, &object_class, (CK_ULONG)sizeof(CK_OBJECT_CLASS)}
	};

	CK_ULONG num_entries = (CK_ULONG)(sizeof(key_obj_template) / sizeof(CK_ATTRIBUTE));

	CK_ULONG key_derivation_label_string_len = 0;
	CK_ULONG key_derivation_context_string_len = 0;

	rv = find_base_key_object(session_, &base_key_obj_handle, key_obj_template, num_entries, &obj_count);
	if(rv == CKR_OK)
	{
		CRYTOOL_INFO << "find_base_key_object success.";
		CK_SP800_108_COUNTER_FORMAT counter_format =
		{
			.bLittleEndian = (CK_BBOOL)CK_FALSE,
			.ulWidthInBits = CK_SP800_108_COUNTER_FORMAT_WIDTH_IN_BITS
		};
		/** Populate the Derived Keying Material length format structure */
		CK_SP800_108_DKM_LENGTH_FORMAT dkm_format =
		{
			.dkmLengthMethod = CK_SP800_108_DKM_LENGTH_SUM_OF_KEYS,
			.bLittleEndian = (CK_BBOOL)CK_FALSE,
			.ulWidthInBits = CK_SP800_REQUIRED_LENGTH_FORMAT_WIDTH
		};

		CK_BYTE prf_params_third_entry[] = {0x00U};

		/** Populate the Pseudo Random Function data structure array */
		CK_PRF_DATA_PARAM prf_data_params[] =
		{
			{
				.type = CK_SP800_108_ITERATION_VARIABLE,
				.pValue = &counter_format,
				.ulValueLen = (CK_ULONG)sizeof(counter_format)
			},
			{
				.type = CK_SP800_108_BYTE_ARRAY,
				.pValue = const_cast<char*> (key_derivation_label_string.data()),
				.ulValueLen = (CK_ULONG)key_derivation_label_string.size()
				// .pValue = label_string,
				// .ulValueLen = (CK_ULONG)sizeof(label_string)
			},
			{
				.type = CK_SP800_108_BYTE_ARRAY,
				.pValue = prf_params_third_entry,
				.ulValueLen = (CK_ULONG)sizeof(prf_params_third_entry)
			},
			{
				.type = CK_SP800_108_BYTE_ARRAY,
				.pValue = const_cast<char*> (key_derivation_context_string.data()),
				.ulValueLen = (CK_ULONG)key_derivation_context_string.size()
				// .pValue = context_string,
				// .ulValueLen = (CK_ULONG)sizeof(context_string)
			},
			{
				.type = CK_SP800_108_DKM_LENGTH,
				.pValue = &dkm_format,
				.ulValueLen = (CK_ULONG)sizeof(dkm_format)
			}
		};
		printf("label_string info string: %s,len:%d \n",prf_data_params[1].pValue,prf_data_params[1].ulValueLen);
		printf("context_string info string: %s,len:%d \n",prf_data_params[3].pValue,prf_data_params[3].ulValueLen);

		// CRYTOOL_INFO << "label_string info string:" <<prf_data_params[1].pValue << "  len:"<<prf_data_params[1].ulValueLen ;
		// CRYTOOL_INFO << "context_string info string:" <<prf_data_params[3].pValue << "  len:"<<prf_data_params[3].ulValueLen ;

		/** Populate Key Derivation Function data structure */
		CK_SP800_108_KDF_PARAMS ck_sp800_108_kdf_params =
		{
			.prfType = CKM_SHA256_HMAC,
			.ulNumberOfDataParams = (CK_ULONG)(sizeof(prf_data_params) / sizeof(prf_data_params[0])),
			.pDataParams = prf_data_params,
			.ulAdditionalDerivedKeys = 0,
			.pAdditionalDerivedKeys = NULL
		};
		CK_MECHANISM mechanism_Derive_key =
		{
			.mechanism = CKM_SP800_108_COUNTER_KDF,
			.pParameter = &ck_sp800_108_kdf_params,
			.ulParameterLen = (CK_ULONG)sizeof(ck_sp800_108_kdf_params)
		};
		/** 32 bytes key ID string "DERIVED_KEY".
		 * PKCS#11 Library shall require that any CKA_ID
		 * generated by the Client Application satisfies the
		 * following constraints:
		 * - a string (array) of CK_BYTEs, maximum 32 bytes padded
		 * with space characters.
		 * - No NULL character.
		 * - Must not start with "NV"
		 * PKCS#11 Library shall return CKR_ATTRIBUTE_VALUE_INVALID
		 * if any of these conditions are not met.
		 */
		CK_BYTE key_id_string[NVPKCS11_MAX_KEY_ID_SIZE];

        if (key_id_input_string.empty() || (key_id_input_string.size() > NVPKCS11_MAX_KEY_ID_SIZE)) {
			CRYTOOL_ERROR << "key_id_input_string is invalid.";
            return CKR_DATA_INVALID;
        }

        rv = string_to_pkcs11_byte_array(const_cast<char*>(key_id_input_string.data()), key_id_input_string.size(), key_id_string, sizeof(key_id_string));
		if(rv != CKR_OK){
			CRYTOOL_ERROR << "convert key_id_input_string to key_id_string failed.";
            return CKR_DATA_INVALID;
		}

		CK_KEY_TYPE key_type = CKK_AES;
		CK_BBOOL sign_attribute;
		CK_BBOOL verify_attribute;
		CK_BBOOL unwrap_attribute;
		CK_MECHANISM_TYPE mechanism_type;

        sign_attribute = (CK_BBOOL)CK_FALSE;
        verify_attribute = (CK_BBOOL)CK_FALSE;
        unwrap_attribute = (CK_BBOOL)CK_FALSE;
		CK_BBOOL encrpto_attribute = (CK_BBOOL)CK_FALSE;
		CK_BBOOL decrpto_attribute = (CK_BBOOL)CK_TRUE;

        // mechanism_type = CKM_AES_GCM;
        mechanism_type = CKM_AES_CBC;
        // mechanism_type = CKM_AES_CBC_PAD;
        // mechanism_type = CKM_AES_CMAC;

        /**
		 * value_length needs to match the size of the wrapping key.
		 * For example if a 16 byte key was used to wrap the data, set
		 * this value to 16 too. (Default value is 32 bytes)
		 */
		CK_ULONG value_length = NVPKCS11_LONG_SECRET_KEY_LENGTH_IN_BYTES;
		// CK_ULONG value_length = NVPKCS11_SECRET_KEY_LENGTH_IN_BYTES;
		
		CK_ATTRIBUTE derive_key_template[] =
		{
			{ CKA_ID, &key_id_string, (CK_ULONG)sizeof(key_id_string) },
			{ CKA_KEY_TYPE, &key_type, (CK_ULONG)sizeof(key_type) },
			{ CKA_VERIFY, &verify_attribute, (CK_ULONG)sizeof(verify_attribute) },
			{ CKA_SIGN, &sign_attribute, (CK_ULONG)sizeof(sign_attribute) },
			{ CKA_UNWRAP, &unwrap_attribute, (CK_ULONG)sizeof(unwrap_attribute) },
			{ CKA_ALLOWED_MECHANISMS, &mechanism_type, (CK_ULONG)sizeof(mechanism_type) },
			{ CKA_CLASS, &object_class, (CK_ULONG)sizeof(CK_OBJECT_CLASS) },
			{ CKA_VALUE_LEN, &value_length, sizeof(CK_ULONG) },
			{ CKA_ENCRYPT, &encrpto_attribute, sizeof(CK_ULONG) },
			{ CKA_DECRYPT, &decrpto_attribute, sizeof(CK_ULONG) }
		};
		CK_ULONG attribute_count = (CK_ULONG)(sizeof(derive_key_template) / sizeof(CK_ATTRIBUTE));

		/**
		 * Derive session Key object. This operation needs to
		 * happen only once and the derived key is persistent
		 * throughout the duration of the session.
		 */

		rv = C_DeriveKey(
				session_,
				&mechanism_Derive_key,
				base_key_obj_handle,
				derive_key_template,
				attribute_count,
				derived_key_handle_ptr);
		if(rv != CKR_OK){
			CRYTOOL_ERROR <<"C_DeriveKey failed. result=" << rv;
		}else{
			CRYTOOL_INFO <<"C_DeriveKey success. result=" << rv;
		}
	}

	return rv;
}


}}}