/*
* Copyright (c) hozonauto. 2021-2021. All rights reserved.
* Description: Http IF class
*/

#ifndef V2C_HTTPLIB_IMPL_CRYPTO_ADAPTER_H
#define V2C_HTTPLIB_IMPL_CRYPTO_ADAPTER_H
#include <memory>
#include <string>
#include <vector>
#include <openssl/rsa.h>
#include "common/entry_point.h"
#include "common/base_id_types.h"
struct x509_st;
// #include "x509_provider.h"
namespace hozon {
namespace netaos {
namespace crypto {
#define RSA_PUB_LEN 1024U
typedef struct {
    uint8_t e[RSA_PUB_LEN];
    uint32_t e_len;
    uint8_t n[RSA_PUB_LEN];
    uint32_t n_len;
} RsaPubKey;

enum EncodeFormat {
    EncodeFormat_Der = 0,
    EncodeFormat_Pem
};

struct DnInfo {
    std::string country;
    std::string organization;
    std::string organization_unit;
    std::string state;
    std::string common_name;
    std::string email_address;
};

enum CryptoAdapterError {
    kNoErr = 0,

    kErrCryptoStart = 50,
    kErrX509CertSignatureFailure = 50,
    kErrX509CertExpired,
    kErrX509CertFuture,
    kErrX509NoIssuerCert,
    kErrX509CertParseErr,
    kErrX509CertStatusUnknown,
    kErrX509CPFault,
};

class CryptoAdapter {
public:
    CryptoAdapter();
    ~CryptoAdapter();
    bool Init(std::string cfg_json);
    bool ImportCert(std::shared_ptr<std::vector<uint8_t>> certdata, EncodeFormat ef);
    int VerifyCert(std::shared_ptr<std::vector<uint8_t>> certdata, EncodeFormat ef);
    // bool ImportCrl(std::shared_ptr<std::vector<uint8_t>> crldata);
    x509::Certificate::Uptrc FindIssuerCert(const x509::X509DN& subject_dn, const x509::X509DN& issuer_dn);
    bool Exist(const x509::Certificate::Uptr& cert);
    void PrintCert(const x509::Certificate& cert);
    bool GetCertInfo(const std::string& cert_path, int64_t& not_before, int64_t& not_after, std::string& cn);
    std::shared_ptr<std::vector<uint8_t>> CreateClientCsr(const cryp::PrivateKey& priv_key, const DnInfo& dn_info);
    std::shared_ptr<std::vector<uint8_t>> CreateClientCsrKeyPair(const std::string &priv_key_slot_uuid_str, const std::string &common_name);
    std::shared_ptr<std::vector<uint8_t>> CreateClientCsrKeyPairWithDn(const std::string &priv_key_slot_uuid_str, const DnInfo &dn_info);
    bool ReadKeyCertFromP12(const std::string& p12_path, const std::string& pass, std::string& pkey_pem, std::string& cert_pem, std::string& ca_chain_pem);
    netaos::crypto::cryp::PrivateKey::Uptrc CreateRsaPrivateKey(std::string slot_uuid);
    RsaPubKey ExportPubKey(std::string slotUuid);
    int ImportPrivateKey(std::vector<uint8_t>& priKey,std::string& slot);
    bool InitKeysProps(std::vector<std::string>& slots);
    // bool SaveKey(const cryp::Key& priv_key, std::string slot);
    bool SavePrivateKey(const netaos::crypto::cryp::PrivateKey& privateKey,const  std::string slotUuid);
    bool SavePublicKey(const netaos::crypto::cryp::PublicKey& publicKey,const std::string slotUuid);
    bool SaveSymmetricKey(const netaos::crypto::cryp::SymmetricKey& symmetricKey,const std::string slotUuid);
    // cryp::Key::Uptrc ReadKey(std::string slot);
    cryp::PrivateKey::Uptrc ReadPrivateKey(const std::string slotUuid);
    int SignWithPrivate(const netaos::crypto::cryp::PrivateKey& key, netaos::crypto::ReadOnlyMemRegion& in, netaos::crypto::ReadWriteMemRegion& out);
    bool VerifyWithPublic(const netaos::crypto::cryp::PublicKey& key, netaos::crypto::ReadOnlyMemRegion in, netaos::crypto::ReadOnlyMemRegion sigdata);
    bool ConvertPemRsaPrivateKey2Der(const std::string& pem, std::vector<uint8_t>& der);
    bool HasKeyInSlot(const std::string& slot_uuid_str);
    bool ClearSlot(std::string& slot_uuid_str);
    bool ExportAndUsePrivate(void* sslctx, std::string priv_slot_uuid);

    bool ReadKeyPair(const std::string& privateKeyFile, const std::string& publicKeyFile, RSA* rsa);
    std::vector<uint8_t> CreateCSR(std::string saveCsrFilePath,DnInfo& dn_info);
    bool IsInit(){
        return inited_;
    };

private:
    netaos::crypto::cryp::PrivateKey::Uptrc CreatePrivateKey(std::string slot_uuid, const netaos::crypto::CryptoAlgId algId,
                                                          const netaos::crypto::AllowedUsageFlags allowedUsage,
                                                          bool isSession = false);
    bool LoadProviders();
    std::shared_ptr<netaos::crypto::Uuid> GetCryptoProviderUuid();
    // bool CommitSlotChange(keys::TransactionScope targetSlots);
    time_t GetTimeNoBefore(x509_st* cert);
    time_t GetTimeNoAfter(x509_st* cert);
    std::string GetSubjectCommonName(x509_st* cert);
    netaos::crypto::cryp::CryptoProvider::Uptr crypto_provider_;
    netaos::crypto::x509::X509Provider::Uptr x509_provider_;
    netaos::crypto::keys::KeyStorageProvider::Uptr keys_provider_;
    std::shared_ptr<netaos::crypto::Uuid> crypto_provider_uuid_;
    std::string cfg_json_;
    static bool log_inited_;
    bool inited_ = false;
};

}
}
}

#endif