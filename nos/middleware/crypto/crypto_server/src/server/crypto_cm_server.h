#pragma once

#include <atomic>
#include <vector>
#include "common/inner_types.h"
#include "cryp/crypto_provider.h"
#include "cryp/cryobj/restricted_use_object.h"
#include "idl/generated/crypto.h"
#include "idl/generated/cryptoPubSubTypes.h"
#include "cm/include/method.h"
#include "crypto_server.h"
namespace hozon {
namespace netaos {
namespace crypto {


class CryptoCmServer {

public:
    static CryptoCmServer& Instance();
    static void Destroy();

    bool Init();
    void Deinit();
    void Stop();
    void Start();

    // void SetCryptoProvider(cryp::CryptoProvider* provider);

    int32_t GenerateKey(int32_t key_type, uint64_t alg_id, uint32_t allowed_usage, bool is_session, bool is_exportable, CryptoKeyRef& key_ref);
    int32_t CreateCipherContext(uint64_t alg_id, int32_t ctx_type, CipherCtxRef& ctx_ref);
    int32_t CryptoTrans(CipherCtxRef ctx_ref, std::vector<uint8_t>& in, std::vector<uint8_t>& out, bool suppress_padding);
    int32_t ContextSetKey(CipherCtxRef ctx_ref, CryptoKeyRef key_ref, uint32_t transform);
    // Release key / context object in server side by specifed ref.
    int32_t ReleaseObject(uint64_t ref);
    int32_t GetPublicKey(CryptoKeyRef private_key_ref, CryptoKeyRef& public_key_ref);
    int32_t LoadKeySlot(std::string keySlotInstanceSpecifier, uint64_t& keySlot_ref);
    int32_t Open(uint64_t keySlot_ref, bool subscribeForUpdates, bool writeable, uint64_t& iOInterface_ref);
    int32_t Save(uint64_t privateKey_ref, uint64_t ioContainer_ref);
    int32_t SaveCopy(uint64_t keySlot_ref, uint64_t ioContainer_ref);
    int32_t LoadPrivateKey(uint64_t ioContainer_ref, CryptoKeyRef& key_ref);
    int32_t BeginTransaction(std::vector<uint64_t> targetSlots, keys::TransactionId& id);
    int32_t CommitTransaction(keys::TransactionId id);

private:
    CryptoCmServer();
    ~CryptoCmServer();

    CryptoKeyRef GetCryptoKeyRefInfo(const cryp::RestrictedUseObject* key);
    CipherCtxRef GetCipherCtxRef(const cryp::CryptoContext* ctx, CipherContextType ctx_type, CryptoTransform transform);
    std::atomic<bool> stopped_{false};
    // cryp::CryptoProvider* crypto_provider_;
    std::recursive_mutex transaction_mutex_;
    // std::shared_ptr<GenerateKeyRequestPubSubType> req_data_type_ = std::make_shared<GenerateKeyRequestPubSubType>();
    // std::shared_ptr<GenerateKeyResultPubSubType> resp_data_type_ = std::make_shared<GenerateKeyResultPubSubType>();
    // GenerateKeyMethodImpl generate_key_method_(req_data_type_,resp_data_type_);

};

}
}
}