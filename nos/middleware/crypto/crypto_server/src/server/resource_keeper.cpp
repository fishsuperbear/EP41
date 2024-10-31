#include "server/resource_keeper.h"

#include <mutex>
#include <memory>

#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {

static ResourceKeeper* sinstance_ = nullptr;
static std::recursive_mutex sinstance_mutex_;
static std::map<int64_t, std::string> res_type_name_mapping_;

ResourceKeeper& ResourceKeeper::Instance()  {

    std::lock_guard<std::recursive_mutex> lock(sinstance_mutex_);
    if (!sinstance_) {
        sinstance_ = new ResourceKeeper();
    }

    return *sinstance_;
}

void ResourceKeeper::Destroy()  {

    std::lock_guard<std::recursive_mutex> lock(sinstance_mutex_);
    if (sinstance_) {
        delete sinstance_;
    }
}

void ResourceKeeper() {
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeSymmetricKey)] = "SymmetricKey";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypePrivateKey)] = "PrivateKey";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypePublicKey)] = "PublicKey";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeSymmetricBlockCipherCtx)] = "SymmetricBlockCipherCtx";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeEncryptorPublicCtx)] = "EncryptorPublicCtx";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeDecryptorPrivateCtx)] = "DecryptorPrivateCtx";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeSignerPrivateCtx)] = "SignerPrivateCtx";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeVerifierPublicCtx)] = "VerifierPublicCtx";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeIoInterfaceContainer)] = "IoInterfaceContainer";
    res_type_name_mapping_[static_cast<uint64_t>(ResourceKeeper::kResourceTypeKeySlot)] = "KeySlot";
}

bool ResourceKeeper::Init()  {


    return false;
}

void ResourceKeeper::Deinit()  {

}

// template < typename T = cryp::SymmetricKey>
cryp::SymmetricKey* ResourceKeeper::QuerySymmetricKey(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeSymmetricKey)) {
        return static_cast<cryp::SymmetricKey*>(resource_map_[ref].res);
    }
    return nullptr;
}

// template < typename T = cryp::PrivateKey>
cryp::PrivateKey* ResourceKeeper::QueryPrivateKey(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypePrivateKey)) {
        return static_cast<cryp::PrivateKey*>(resource_map_[ref].res);
    }
    return nullptr;
}

// template < typename T = cryp::PublicKey>
cryp::PublicKey* ResourceKeeper::QueryPublicKey(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    CRYP_ERROR << "QueryPublicKey: "<<ref ;
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypePublicKey)) {
        return static_cast<cryp::PublicKey*>(resource_map_[ref].res);
    }
    return nullptr;
}

// template < typename T = cryp::SymmetricBlockCipherCtx>
cryp::SymmetricBlockCipherCtx* ResourceKeeper::QuerySymmetricBlockCipherCtx(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeSymmetricBlockCipherCtx)) {
        return static_cast<cryp::SymmetricBlockCipherCtx*>(resource_map_[ref].res);
    }
    return nullptr;
}

cryp::EncryptorPublicCtx* ResourceKeeper::QueryEncryptorPublicCtx(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeEncryptorPublicCtx)) {
        return static_cast<cryp::EncryptorPublicCtx*>(resource_map_[ref].res);
    }
    return nullptr;
}

cryp::DecryptorPrivateCtx* ResourceKeeper::QueryDecryptorPrivateCtx(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeDecryptorPrivateCtx)) {
        return static_cast<cryp::DecryptorPrivateCtx*>(resource_map_[ref].res);
    }
    return nullptr;
}

cryp::SignerPrivateCtx* ResourceKeeper::QuerySignerPrivateCtx(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeSignerPrivateCtx)) {
        return static_cast<cryp::SignerPrivateCtx*>(resource_map_[ref].res);
    }
    return nullptr;
}

cryp::VerifierPublicCtx* ResourceKeeper::QueryVerifierPublicCtx(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end()) &&
        (resource_map_[ref].type == kResourceTypeVerifierPublicCtx)) {
        return static_cast<cryp::VerifierPublicCtx*>(resource_map_[ref].res);
    }
    return nullptr;
}

IOInterface* ResourceKeeper::QueryIoInterfaceContainer(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeIoInterfaceContainer)) {
        return static_cast<IOInterface*>(resource_map_[ref].res);
    }
    return nullptr;
}

keys::KeySlot* ResourceKeeper::QueryKeySlot(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeKeySlot)) {
        return static_cast<keys::KeySlot*>(resource_map_[ref].res);
    }
    return nullptr;
}

x509::CertSignRequest* ResourceKeeper::QueryCertSignRequest(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeCertSignRequest)) {
        return static_cast<x509::CertSignRequest*>(resource_map_[ref].res);
    }
    return nullptr;
}

x509::X509DN* ResourceKeeper::QueryX509Dn(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if ((resource_map_.find(ref) != resource_map_.end())
        && (resource_map_[ref].type == kResourceTypeX509DnRequest)) {
        return static_cast<x509::X509DN*>(resource_map_[ref].res);
    }
    return nullptr;
}


// template < typename T = cryp::SymmetricKey>
uint64_t ResourceKeeper::KeepSymmetricKey(cryp::SymmetricKey* res) {
    return KeepResource(kResourceTypeSymmetricKey, res);
}

// template < typename T = cryp::PrivateKey>
uint64_t ResourceKeeper::KeepPrivateKey(cryp::PrivateKey* res) {
    return KeepResource(kResourceTypePrivateKey, res);
}

// template < typename T = cryp::PublicKey>
uint64_t ResourceKeeper::KeepPublicKey(cryp::PublicKey* res) {
    return KeepResource(kResourceTypePublicKey, res);
}

// template < typename T = cryp::SymmetricBlockCipherCtx>
uint64_t ResourceKeeper::KeepSymmetricBlockCipherCtx(cryp::SymmetricBlockCipherCtx* res) {
    return KeepResource(kResourceTypeSymmetricBlockCipherCtx, res);
}

uint64_t ResourceKeeper::KeepEncryptorPublicCtx(cryp::EncryptorPublicCtx* res) {
    return KeepResource(kResourceTypeEncryptorPublicCtx, res);
}

uint64_t ResourceKeeper::KeepDecryptorPrivateCtx(cryp::DecryptorPrivateCtx* res) {
    return KeepResource(kResourceTypeDecryptorPrivateCtx, res);
}

uint64_t ResourceKeeper::KeepSignerPrivateCtx(cryp::SignerPrivateCtx* res) {
    return KeepResource(kResourceTypeSignerPrivateCtx, res);
}

uint64_t ResourceKeeper:: KeepIoInterfaceContainer(IOInterface* res) {
    return KeepResource(kResourceTypeIoInterfaceContainer, res);
}

uint64_t ResourceKeeper::KeepKeySlot(keys::KeySlot* res) {
    return KeepResource(kResourceTypeKeySlot, res);
}

uint64_t ResourceKeeper::KeepCertSignRequest(x509::CertSignRequest* res) {
    return KeepResource(kResourceTypeCertSignRequest, res);
}

uint64_t ResourceKeeper::KeepX509DN(x509::X509DN* res) {
    return KeepResource(kResourceTypeX509DnRequest, res);
}

uint64_t ResourceKeeper::KeepVerifierPublicCtx(cryp::VerifierPublicCtx* res) {
    return KeepResource(kResourceTypeVerifierPublicCtx, res);
}

void ResourceKeeper::Release(uint64_t ref) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    if (resource_map_.find(ref) != resource_map_.end()) {
        return;
    }

    std::string type_name = res_type_name_mapping_[resource_map_[ref].type];
    void* p = reinterpret_cast<void *>(resource_map_[ref].res);

    switch (resource_map_[ref].type) {
    case kResourceTypeSymmetricKey:
        delete static_cast<cryp::SymmetricKey*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypePrivateKey:
        delete static_cast<cryp::PrivateKey*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypePublicKey:
        delete static_cast<cryp::PublicKey*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypeSymmetricBlockCipherCtx:
        delete static_cast<cryp::SymmetricBlockCipherCtx*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypeEncryptorPublicCtx:
        delete static_cast<cryp::EncryptorPublicCtx*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypeDecryptorPrivateCtx:
        delete static_cast<cryp::DecryptorPrivateCtx*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypeSignerPrivateCtx:
        delete static_cast<cryp::SignerPrivateCtx*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypeIoInterfaceContainer:
        delete static_cast<IOInterface*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypeKeySlot:
        delete static_cast<keys::KeySlot*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    case kResourceTypeVerifierPublicCtx:
        delete static_cast<cryp::VerifierPublicCtx*>(resource_map_[ref].res);
        resource_map_.erase(ref);
    break;
    default:
        CRYP_ERROR << "Unknow type. Cannot correctly release resource.\n";
        return;
    break;
    }

    CRYP_INFO << "Object of type " << type_name << "is released. addr: 0x" << p;
}

ResourceKeeper::ResourceKeeper() {

}

ResourceKeeper::~ResourceKeeper() {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    for (auto it : resource_map_) {
        Release(it.first);
    }
}

uint64_t ResourceKeeper::KeepResource(int32_t type, void* res) {
    std::lock_guard<std::recursive_mutex> lock(resource_mutex_);
    CRYP_INFO << "Keep resource. type: " << type << ", res: " << res;
    if (resource_map_.find(reinterpret_cast<uint64_t>(res)) != resource_map_.end()) {
        CRYP_ERROR << "Resource already exists. Please check whether duplicated keep operation or not correctly released.";
        return 0x0u;
    }

    resource_map_[reinterpret_cast<uint64_t>(res)] = {type, res};
    return reinterpret_cast<uint64_t>(res);
}

}
}
}