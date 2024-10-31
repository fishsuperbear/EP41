#include "crypto_cm_client.h"
#include <mutex>
#include <memory>
#include <chrono>
#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"
#include "common/type_converter.h"
#include "keys/cimpl_keyslot.h"

namespace hozon {
namespace netaos {
namespace crypto {

template<class T>
void DeleteMethod(T *pt){
    if(pt){
        pt->Deinit();
    }
};

const int64_t CRYPTO_CM_REQUEST_TIMEOUT = 1000;
const uint32_t CRYPTO_DOMAIN = 2;
CryptoCmClient *CryptoCmClient::sinstance_ = nullptr;
static std::mutex sinstance_mutex_;

CryptoCmClient& CryptoCmClient::Instance()  {
    std::lock_guard<std::mutex> lock(sinstance_mutex_);
    if (!CryptoCmClient::sinstance_) {
        CryptoCmClient::sinstance_ = new CryptoCmClient();
        CRYP_INFO << "CryptoCmClient construct called.";
    }

    return *CryptoCmClient::sinstance_;
}

bool CryptoCmClient::Init()  {

    bool ret = false;
    // Method for [generate key].

    if (!stopped_) {
        if (0 == gen_key_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:GenerateKeyMethod success.";
            ret = true;
        }else{
            ret = false;
        }
    }
    
    // Method for [create cipher context].

    if (!stopped_) {
        if (0 == create_cipher_context_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:CreateCipherContextMethod success." ;
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    // Method for [crypto].
    if (!stopped_) {
        if (0 == crypto_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:CryptoMethod success." ;
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    // Method for [crypto].
    if (!stopped_) {
        if (0 == context_set_key_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:ContextSetKeyMethod success." ;
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    // Method for [get public key].
    if (!stopped_) {
        if (0 == get_publickey_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:GetPublicKeyMethod success." ;
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    // Method for [get load keySlot].
    if (!stopped_) {
        if (0 == load_keyslot_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:LoadKeySlotMethod success." ;
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    // Method for [get open keySlot].
    if (!stopped_) {
        if (0 == open_keyslot_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:OpenKeySlotMethod success." ;
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    if (!stopped_) {
        if (0 == save_container_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:SaveContainerMethod success.";
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    if (!stopped_) {
        if (0 == save_copy_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:SaveCopyMethod success.";
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }
    
    if (!stopped_) {
        if (0 == load_private_key_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:LoadPrivateKeyMethod success." ;
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }
    
    if (!stopped_) {
        if (0 == begin_transaction_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:BeginTransactionMethod success.";
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    if (!stopped_) {
        if (0 == commit_transaction_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:CommitTransactionMethod success.";
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    // Method for [release object].
    if (!stopped_) {
        if (0 == release_object_method_->WaitServiceOnline(1000)) {
            CRYP_INFO << "find service:ReleaseObjectMethod success.";
            ret = !(ret^true);
        }else{
            ret = false;
        }
    }

    inited_ = ret;
    return ret;
}

bool CryptoCmClient::Deinit()  {
    Stop();
    if(gen_key_method_.get())gen_key_method_->Deinit();
    if(create_cipher_context_method_.get())create_cipher_context_method_->Deinit();
    if(crypto_method_.get())crypto_method_->Deinit();
    if(context_set_key_method_.get())context_set_key_method_->Deinit();
    if(release_object_method_.get())release_object_method_->Deinit();
    if(get_publickey_method_.get())get_publickey_method_->Deinit();
    if(load_keyslot_method_.get())load_keyslot_method_->Deinit();
    if(open_keyslot_method_.get())open_keyslot_method_->Deinit();
    if(save_container_method_.get())save_container_method_->Deinit();
    if(save_copy_method_.get())save_copy_method_->Deinit();
    if(load_private_key_method_.get())load_private_key_method_->Deinit();
    if(begin_transaction_method_.get())begin_transaction_method_->Deinit();
    if(commit_transaction_method_.get())commit_transaction_method_->Deinit();
    return true;
}

void CryptoCmClient::Stop() {
    stopped_ = true;
}

int32_t CryptoCmClient::GenerateKey(int32_t key_type, uint64_t alg_id, uint32_t allowed_usage, bool is_session, bool is_exportable, CryptoKeyRef& key_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::GenerateKey begin.";

    auto gen_key_request = std::make_shared<GenerateKeyRequest>();
    gen_key_request->key_type() = key_type;
    gen_key_request->alg_id() = alg_id;
    gen_key_request->allowed_usage() = allowed_usage;
    gen_key_request->is_session() = is_session;
    gen_key_request->is_exportable() = is_exportable;
    gen_key_request->is_exportable() = is_exportable;
    gen_key_request->fire_forget(false);
    auto gen_key_result = std::make_shared<GenerateKeyResult>();
    int cm_res = gen_key_method_->Request(gen_key_request, gen_key_result, 8000);

    if (cm_res != 0) {
        CRYP_ERROR<<"gen_key_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }else{
        CRYP_INFO<<"gen_key_method_->Request success.";
    }
    CRYP_INFO<<"gen_key_result uid:"<<gen_key_result->key().crypto_object_info().object_uid();
    TypeConverter::CmStructToInnerType(gen_key_result->key(), key_ref);
    return static_cast<int32_t>(gen_key_result->code());
}

int32_t CryptoCmClient::CreateCipherContext(uint64_t alg_id, int32_t ctx_type, CipherCtxRef& ctx_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::CreateCipherContext begin.";
    auto create_cipher_context_request = std::make_shared<CreateCipherContextRequest>();
    create_cipher_context_request->alg_id() = alg_id;
    create_cipher_context_request->ctx_type() = ctx_type;
    create_cipher_context_request->fire_forget(false);
    auto create_cipher_context_result = std::make_shared<CreateCipherContextResult>();
    int cm_res = create_cipher_context_method_->Request(create_cipher_context_request, create_cipher_context_result, CRYPTO_CM_REQUEST_TIMEOUT);

    if (cm_res != 0) {
        CRYP_ERROR<<"create_cipher_context_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }

    TypeConverter::CmStructToInnerType(create_cipher_context_result->ctx_ref(), ctx_ref);
    return static_cast<int32_t>(create_cipher_context_result->code());
}

int32_t CryptoCmClient::CryptoTrans(CipherCtxRef ctx_ref, std::vector<uint8_t>& in, std::vector<uint8_t>& out, bool suppress_padding) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }

    auto crypto_request = std::make_shared<CryptoRequest>();
    crypto_request->ctx_ref().alg_id() = ctx_ref.alg_id;
    crypto_request->ctx_ref().ref() = ctx_ref.ref;
    crypto_request->ctx_ref().ctx_type() = ctx_ref.ctx_type;
    crypto_request->input() = in;
    if (ctx_ref.ctx_type == kCipherContextType_VerifierPublic) {
        crypto_request->input1() = out;
    }
    
    crypto_request->suppress_padding() = suppress_padding;
    auto crypto_result = std::make_shared<CryptoResult>();
    CRYP_INFO<<"crypto_request->input size: "<<crypto_request->input().size();
    crypto_request->fire_forget(false);

    int cm_res = crypto_method_->Request(crypto_request, crypto_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"crypto_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }
    CRYP_INFO << "Crypto trans out(" << crypto_result->output().size()
                  << "):\n" << hozon::netaos::crypto::CryptoLogger::GetInstance().ToHexString(crypto_result->output().data(), crypto_result->output().size());
    out = crypto_result->output();
    // return static_cast<int32_t>(crypto_result->code());
    return 0;
}

int32_t CryptoCmClient::ContextSetKey(CipherCtxRef ctx_ref, CryptoKeyRef key_ref, uint32_t transform) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::ContextSetKey begin.";
    auto context_set_key_request = std::make_shared<ContextSetKeyRequest>();
    context_set_key_request->ctx_ref().alg_id() = ctx_ref.alg_id;
    context_set_key_request->ctx_ref().ref() = ctx_ref.ref;
    context_set_key_request->ctx_ref().ctx_type() = ctx_ref.ctx_type;
    context_set_key_request->key_ref().alg_id() = key_ref.alg_id;
    context_set_key_request->key_ref().ref() = key_ref.ref;
    context_set_key_request->transform() = transform;
    auto context_set_key_result = std::make_shared<ContextSetKeyResult>();
    
    int cm_res = context_set_key_method_->Request(context_set_key_request, context_set_key_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"context_set_key_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }else{
        CRYP_INFO<<"context_set_key_method_->Request success.";
    }

    return static_cast<int32_t>(context_set_key_result->code());
}

int32_t CryptoCmClient::ReleaseObject(uint64_t ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::ReleaseObject begin.";
    auto release_object_request = std::make_shared<ReleaseObjectRequest>();
    release_object_request->ref() = ref;
    release_object_request->fire_forget(false);
    auto release_object_result = std::make_shared<ReleaseObjectResult>();
    int cm_res = release_object_method_->Request(release_object_request, release_object_result, CRYPTO_CM_REQUEST_TIMEOUT);

    if (cm_res != 0) {
        CRYP_ERROR<<"release_object_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }

    return static_cast<int32_t>(release_object_result->code());
}

int32_t CryptoCmClient::GetPublicKey(CryptoKeyRef private_key_ref, CryptoKeyRef& public_key_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::GetPublicKey begin.";
    auto getPublicKeyFromPrivateKeyRequest = std::make_shared<GetPublicKeyFromPrivateKeyRequest>();
    TypeConverter::InnerTypeToCmStruct(private_key_ref, getPublicKeyFromPrivateKeyRequest->private_key_ref());

    getPublicKeyFromPrivateKeyRequest->fire_forget(false);
    auto generateKey_result = std::make_shared<GenerateKeyResult>();
    int cm_res = get_publickey_method_->Request(getPublicKeyFromPrivateKeyRequest, generateKey_result, CRYPTO_CM_REQUEST_TIMEOUT);
    TypeConverter::CmStructToInnerType(generateKey_result->key(), public_key_ref);
    if (cm_res != 0) {
        CRYP_ERROR<<"get_publickey_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }

    // TypeConverter::CmStructToInnerType(getPublicKeyFromPrivateKeyRequest->ctx_ref(), ctx_ref);
    return static_cast<int32_t>(generateKey_result->code());
}

int32_t CryptoCmClient::LoadKeySlot(std::string keySlotInstanceSpecifier, CryptoSlotRef& keySlot_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::LoadKeySlot begin.";
    auto loadKeySlot_request = std::make_shared<LoadKeySlotRequest>();
    loadKeySlot_request->keySlotInstanceSpecifier() = keySlotInstanceSpecifier;
    auto loadKeySlot_result = std::make_shared<LoadKeySlotResult>();
    // CRYP_INFO<<"loadKeySlot_request->input size: "<<loadKeySlot_request->input().size();
    loadKeySlot_request->fire_forget(false);

    int cm_res = load_keyslot_method_->Request(loadKeySlot_request, loadKeySlot_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"load_keyslot_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }
    keySlot_ref.ref = loadKeySlot_result->keySlot_ref();
    CRYP_INFO<<"crypto_result : "<<loadKeySlot_result->keySlot_ref();
    return static_cast<int32_t>(loadKeySlot_result->code());
}

int32_t CryptoCmClient::Open(uint64_t keySlot_ref, bool subscribeForUpdates, bool writeable, uint64_t& iOInterface_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::Open begin.";
    auto openKeySlot_request = std::make_shared<OpenKeySlotRequest>();
    openKeySlot_request->keySlot_ref() = keySlot_ref;
    openKeySlot_request->subscribeForUpdates() = subscribeForUpdates;
    openKeySlot_request->writeable() = writeable;

    auto openKeySlot_result = std::make_shared<OpenKeySlotResult>();
    // CRYP_INFO<<"openKeySlot_request->input size: "<<openKeySlot_request->input().size();
    openKeySlot_request->fire_forget(false);

    int cm_res = open_keyslot_method_->Request(openKeySlot_request, openKeySlot_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"open_keyslot_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }
    iOInterface_ref = openKeySlot_result->iOInterface_ref();
    CRYP_INFO<<"crypto_result : "<<openKeySlot_result->iOInterface_ref();
    return static_cast<int32_t>(openKeySlot_result->code());
}

int32_t CryptoCmClient::Save(uint64_t privateKey_ref, uint64_t ioContainer_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::Save begin.";
    auto saveContainer_request = std::make_shared<SaveContainerRequest>();
    saveContainer_request->key_ref().ref() = privateKey_ref;
    saveContainer_request->iOInterface_ref() = ioContainer_ref;
    // saveContainer_request->contentProps() = cmKeySlotContentProps;
    auto saveContainer_result = std::make_shared<SaveContainerResult>();
    // CRYP_INFO<<"saveContainer_request->input size: "<<saveContainer_request->input().size();
    saveContainer_request->fire_forget(true);

    int cm_res = save_container_method_->Request(saveContainer_request, saveContainer_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"save_container_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }
    // iOInterface_ref = saveContainer_result->iOInterface_ref();
    CRYP_INFO<<"crypto_result : "<<saveContainer_result->code();
    return static_cast<int32_t>(saveContainer_result->code());
}

int32_t CryptoCmClient::SaveCopy(uint64_t keySlot_ref, uint64_t ioContainer_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::SaveCopy begin.";
    auto saveCopy_request = std::make_shared<SaveCopyRequest>();
    saveCopy_request->keySlot_ref() = keySlot_ref;
    saveCopy_request->iOInterface_ref() = ioContainer_ref;
    // saveCopy_request->contentProps() = cmKeySlotContentProps;
    auto saveCopy_result = std::make_shared<SaveCopyResult>();
    // CRYP_INFO<<"saveCopy_request->input size: "<<saveCopy_request->input().size();
    saveCopy_request->fire_forget(true);

    int cm_res = save_copy_method_->Request(saveCopy_request, saveCopy_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"save_copy_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }
    // iOInterface_ref = saveCopy_result->iOInterface_ref();
    CRYP_INFO<<"crypto_result : "<<saveCopy_result->code();
    return static_cast<int32_t>(saveCopy_result->code());
}

int32_t CryptoCmClient::LoadPrivateKey(uint64_t ioContainer_ref, CryptoKeyRef& key_ref) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::LoadPrivateKey begin.";
    auto load_private_key_request = std::make_shared<LoadPrivateKeyRequest>();
    load_private_key_request->iOInterface_ref() = ioContainer_ref;
    auto load_private_key_result = std::make_shared<LoadPrivateKeyResult>();

    int cm_res = load_private_key_method_->Request(load_private_key_request, load_private_key_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"load_private_key_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }else{
        CRYP_INFO<<"load_private_key_method_->Request success.";
    }
    CRYP_INFO<<"load_private_key_method_ uid:"<<load_private_key_result->key().crypto_object_info().object_uid();
    TypeConverter::CmStructToInnerType(load_private_key_result->key(), key_ref);
    return static_cast<int32_t>(load_private_key_result->code());
}

int32_t CryptoCmClient::BeginTransaction(const keys::TransactionScope& targetSlots, keys::TransactionId& id) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::BeginTransaction begin.";
    auto beginTransaction_request = std::make_shared<BeginTransactionRequest>();

    for (auto slot : targetSlots) {
        keys::KeySlot* slot_ptr = slot;
        auto ptr = dynamic_cast<keys::CimplKeySlot*>(slot_ptr);
        if (ptr) {
            beginTransaction_request->transactionScope().push_back(ptr->getSlotRef().ref);
        } else {
            CRYP_ERROR<<"BeginTransaction: there is a inVaild slot ";
        }
    }
    // beginTransaction_request->contentProps() = cmKeySlotContentProps;
    auto beginTransaction_result = std::make_shared<BeginTransactionResult>();
    // CRYP_INFO<<"beginTransaction_request->input size: "<<beginTransaction_request->input().size();
    beginTransaction_request->fire_forget(false);

    int cm_res = begin_transaction_method_->Request(beginTransaction_request, beginTransaction_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"begin_transaction_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }
    // iOInterface_ref = beginTransaction_result->iOInterface_ref();
    CRYP_INFO<<"crypto_result : "<<beginTransaction_result->code();
    id = beginTransaction_result->transactionId();
    return static_cast<int32_t>(beginTransaction_result->code());
}

int32_t CryptoCmClient::CommitTransaction(keys::TransactionId id) {
    if (!inited_) {
        return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
    }
    CRYP_INFO<<"CryptoCmClient::CommitTransaction begin.";
    auto commitTransaction_request = std::make_shared<CommitTransactionRequest>();
    commitTransaction_request->transactionId() = static_cast<uint64_t>(id);

    // commitTransaction_request->contentProps() = cmKeySlotContentProps;
    auto commitTransaction_result = std::make_shared<CommitTransactionResult>();
    // CRYP_INFO<<"commitTransaction_request->input size: "<<commitTransaction_request->input().size();
    commitTransaction_request->fire_forget(false);

    int cm_res = commit_transaction_method_->Request(commitTransaction_request, commitTransaction_result, CRYPTO_CM_REQUEST_TIMEOUT);
    if (cm_res != 0) {
        CRYP_ERROR<<"commit_transaction_method_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }
    // iOInterface_ref = commitTransaction_result->iOInterface_ref();
    CRYP_INFO<<"crypto_result : "<<commitTransaction_result->code();
    return static_cast<int32_t>(commitTransaction_result->code());
}

CryptoCmClient::CryptoCmClient() {
    // Method for [generate key].
    auto gen_key_req_ps_type = std::make_shared<GenerateKeyRequestPubSubType>();
    auto gen_key_res_ps_type = std::make_shared<GenerateKeyResultPubSubType>();
    gen_key_method_.reset(new GenerateKeyMethod(gen_key_req_ps_type, gen_key_res_ps_type));
    gen_key_method_->Init(CRYPTO_DOMAIN, "GenerateKeyRequest");
    
    // Method for [create cipher context].
    auto create_ciphe_context_req_ps_type = std::make_shared<CreateCipherContextRequestPubSubType>();
    auto create_ciphe_context_res_ps_type = std::make_shared<CreateCipherContextResultPubSubType>();
    create_cipher_context_method_.reset(
        new CreateCipherContextMethod(create_ciphe_context_req_ps_type, create_ciphe_context_res_ps_type));
    create_cipher_context_method_->Init(CRYPTO_DOMAIN, "CreateCipherContextRequest");
   

    // Method for [crypto].
    auto crypto_req_ps_type = std::make_shared<CryptoRequestPubSubType>();
    auto crypto_res_ps_type = std::make_shared<CryptoResultPubSubType>();
    crypto_method_.reset(new CryptoMethod(crypto_req_ps_type, crypto_res_ps_type));
    crypto_method_->Init(CRYPTO_DOMAIN, "CryptoRequest");
    
    // Method for [crypto].
    auto context_set_key_req_ps_type = std::make_shared<ContextSetKeyRequestPubSubType>();
    auto context_set_key_res_ps_type = std::make_shared<ContextSetKeyResultPubSubType>();
    context_set_key_method_.reset(new ContextSetKeyMethod(context_set_key_req_ps_type, context_set_key_res_ps_type));
    context_set_key_method_->Init(CRYPTO_DOMAIN, "ContextSetKeyRequest");
   

    // Method for [get public key].
    auto get_publickey_req_ps_type = std::make_shared<GetPublicKeyFromPrivateKeyRequestPubSubType>();
    auto get_publickey_res_ps_type = std::make_shared<GenerateKeyResultPubSubType>();
    get_publickey_method_.reset(new GetPublicKeyMethod(get_publickey_req_ps_type, get_publickey_res_ps_type));
    get_publickey_method_->Init(CRYPTO_DOMAIN, "GetPublicKey");
   

    // Method for [get load keySlot].
    auto load_keyslot_req_ps_type = std::make_shared<LoadKeySlotRequestPubSubType>();
    auto load_keyslot_res_ps_type = std::make_shared<LoadKeySlotResultPubSubType>();
    load_keyslot_method_.reset(new LoadKeySlotMethod(load_keyslot_req_ps_type, load_keyslot_res_ps_type));
    load_keyslot_method_->Init(CRYPTO_DOMAIN, "LoadKeySlot");
    
    // Method for [get open keySlot].
    auto open_keyslot_req_ps_type = std::make_shared<OpenKeySlotRequestPubSubType>();
    auto open_keyslot_res_ps_type = std::make_shared<OpenKeySlotResultPubSubType>();
    open_keyslot_method_.reset(new OpenKeySlotMethod(open_keyslot_req_ps_type, open_keyslot_res_ps_type));
    open_keyslot_method_->Init(CRYPTO_DOMAIN, "OpenKeySlot");

    auto save_container_req_ps_type = std::make_shared<SaveContainerRequestPubSubType>();
    auto save_container_res_ps_type = std::make_shared<SaveContainerResultPubSubType>();
    save_container_method_.reset(new SaveContainerMethod(save_container_req_ps_type, save_container_res_ps_type));
    save_container_method_->Init(CRYPTO_DOMAIN, "SaveContainer");

    auto save_copy_req_ps_type = std::make_shared<SaveCopyRequestPubSubType>();
    auto save_copy_res_ps_type = std::make_shared<SaveCopyResultPubSubType>();
    save_copy_method_.reset(new SaveCopyMethod(save_copy_req_ps_type, save_copy_res_ps_type));
    save_copy_method_->Init(CRYPTO_DOMAIN, "SaveCopy");

    auto load_private_key_req_ps_type = std::make_shared<LoadPrivateKeyRequestPubSubType>();
    auto load_private_key_res_ps_type = std::make_shared<LoadPrivateKeyResultPubSubType>();
    load_private_key_method_.reset(
        new LoadPrivateKeyMethod(load_private_key_req_ps_type, load_private_key_res_ps_type));
    load_private_key_method_->Init(CRYPTO_DOMAIN, "LoadPrivateKey");

    auto begin_transaction_req_ps_type = std::make_shared<BeginTransactionRequestPubSubType>();
    auto begin_transaction_res_ps_type = std::make_shared<BeginTransactionResultPubSubType>();
    begin_transaction_method_.reset(
        new BeginTransactionMethod(begin_transaction_req_ps_type, begin_transaction_res_ps_type));
    begin_transaction_method_->Init(CRYPTO_DOMAIN, "BeginTransaction");

    auto commit_transaction_req_ps_type = std::make_shared<CommitTransactionRequestPubSubType>();
    auto commit_transaction_res_ps_type = std::make_shared<CommitTransactionResultPubSubType>();
    commit_transaction_method_.reset(
        new CommitTransactionMethod(commit_transaction_req_ps_type, commit_transaction_res_ps_type));
    commit_transaction_method_->Init(CRYPTO_DOMAIN, "CommitTransaction");

    // Method for [release object].
    auto release_object_req_ps_type = std::make_shared<ReleaseObjectRequestPubSubType>();
    auto release_object_res_ps_type = std::make_shared<ReleaseObjectResultPubSubType>();
    release_object_method_.reset(new ReleaseObjectMethod(release_object_req_ps_type, release_object_res_ps_type));
    release_object_method_->Init(CRYPTO_DOMAIN, "ReleaseObjectRequest");
}

CryptoCmClient::~CryptoCmClient() {
    // Stop();
    Deinit();
}

}
}
}