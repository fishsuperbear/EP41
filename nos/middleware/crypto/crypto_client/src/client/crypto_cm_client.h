#pragma once

#include <atomic>
#include <vector>
#include <common/inner_types.h>
#include "cm/include/method.h"
#include "idl/generated/crypto.h"
#include "idl/generated/cryptoPubSubTypes.h"
#include "cryp/cryobj/crypto_object.h"
#include "keys/key_slot_prototype_props.h"
#include "keys/key_slot_content_props.h"
#include "keys/elementary_types.h"
namespace hozon {
namespace netaos {
namespace crypto {
extern  const uint32_t CRYPTO_DOMAIN;
class CryptoCmClient {

public:

    using GenerateKeyMethod = hozon::netaos::cm::Client<GenerateKeyRequest, GenerateKeyResult>;
    using CreateCipherContextMethod = hozon::netaos::cm::Client<CreateCipherContextRequest, CreateCipherContextResult>;
    using CryptoMethod = hozon::netaos::cm::Client<CryptoRequest, CryptoResult>;
    using ContextSetKeyMethod = hozon::netaos::cm::Client<ContextSetKeyRequest, ContextSetKeyResult>;
    using ReleaseObjectMethod = hozon::netaos::cm::Client<ReleaseObjectRequest, ReleaseObjectResult>;
    using GetPublicKeyMethod = hozon::netaos::cm::Client<GetPublicKeyFromPrivateKeyRequest, GenerateKeyResult>;
    using LoadKeySlotMethod = hozon::netaos::cm::Client<LoadKeySlotRequest, LoadKeySlotResult>;
    using OpenKeySlotMethod = hozon::netaos::cm::Client<OpenKeySlotRequest, OpenKeySlotResult>;
    using SaveContainerMethod = hozon::netaos::cm::Client<SaveContainerRequest, SaveContainerResult>;
    using SaveCopyMethod = hozon::netaos::cm::Client<SaveCopyRequest, SaveCopyResult>;
    using LoadPrivateKeyMethod = hozon::netaos::cm::Client<LoadPrivateKeyRequest, LoadPrivateKeyResult>;
    using BeginTransactionMethod = hozon::netaos::cm::Client<BeginTransactionRequest, BeginTransactionResult>;
    using CommitTransactionMethod = hozon::netaos::cm::Client<CommitTransactionRequest, CommitTransactionResult>;

    static CryptoCmClient& Instance();
    static void Destroy();

    bool Init();
    bool Deinit();
    void Stop();

    int32_t GenerateKey(int32_t key_type, uint64_t alg_id, uint32_t allowed_usage, bool is_session, bool is_exportable, CryptoKeyRef& key_ref);
    int32_t CreateCipherContext(uint64_t alg_id, int32_t ctx_type, CipherCtxRef& ctx_ref);
    int32_t CryptoTrans(CipherCtxRef ctx_ref, std::vector<uint8_t>& in, std::vector<uint8_t>& out, bool suppress_padding);
    int32_t ContextSetKey(CipherCtxRef ctx_ref, CryptoKeyRef key_ref, uint32_t transform);

    int32_t ContextSetKey(CryptoKeyRef key_ref);
    // Release key / context object in server side by specifed ref.
    int32_t ReleaseObject(uint64_t ref);

    int32_t GetPublicKey(CryptoKeyRef private_key_ref, CryptoKeyRef& public_key_ref);

    int32_t LoadKeySlot(std::string keySlotInstanceSpecifier, CryptoSlotRef& keySlot_ref);

    int32_t Open(uint64_t keySlot_ref, bool subscribeForUpdates, bool writeable, uint64_t& iOInterface_ref);

    int32_t Save(uint64_t privateKey_ref, uint64_t ioContainer_ref);

    int32_t SaveCopy(uint64_t keySlot_ref, uint64_t ioContainer_ref);

    int32_t BeginTransaction(const keys::TransactionScope& targetSlots, keys::TransactionId& id);

    int32_t CommitTransaction(keys::TransactionId id);

    int32_t LoadPrivateKey(uint64_t ioContainer_ref, CryptoKeyRef& key_ref);

private:
    CryptoCmClient();
    ~CryptoCmClient();

    std::atomic<bool> inited_{false};
    std::atomic<bool> stopped_{false};

    std::unique_ptr<GenerateKeyMethod> gen_key_method_ = nullptr;
    std::unique_ptr<CreateCipherContextMethod> create_cipher_context_method_ = nullptr;
    std::unique_ptr<CryptoMethod> crypto_method_ = nullptr;
    std::unique_ptr<ContextSetKeyMethod> context_set_key_method_ = nullptr;
    std::unique_ptr<ReleaseObjectMethod> release_object_method_ = nullptr;
    std::unique_ptr<GetPublicKeyMethod> get_publickey_method_ = nullptr;
    std::unique_ptr<LoadKeySlotMethod> load_keyslot_method_ = nullptr;
    std::unique_ptr<OpenKeySlotMethod> open_keyslot_method_ = nullptr;
    std::unique_ptr<SaveContainerMethod> save_container_method_ = nullptr;
    std::unique_ptr<SaveCopyMethod> save_copy_method_ = nullptr;
    std::unique_ptr<LoadPrivateKeyMethod> load_private_key_method_ = nullptr;
    std::unique_ptr<BeginTransactionMethod> begin_transaction_method_ = nullptr;
    std::unique_ptr<CommitTransactionMethod> commit_transaction_method_ = nullptr;
    static CryptoCmClient* sinstance_;

};

}
}
}