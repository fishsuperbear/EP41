#include "cimpl_keyslot.h"

#include <memory>
#include <string>
#include <vector>
#include <ios>
#include <fstream>
#include <iostream>

#include "cimpl_io_interface.h"
#include "common/crypto_logger.hpp"
#include "cryp/cimpl_crypto_provider.h"
#include "client/crypto_cm_client.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

CimplKeySlot::CimplKeySlot(const CryptoSlotRef& keySlot_ref)
: keySlot_ref_(keySlot_ref) {

}

CimplKeySlot::~CimplKeySlot() {
    CryptoCmClient::Instance().ReleaseObject(keySlot_ref_.ref);
}

netaos::core::Result<void> CimplKeySlot::Clear() noexcept{
    contentProps_.mAlgId = crypto::kAlgIdUndefined;
    contentProps_.mObjectSize = 0;
    contentProps_.mObjectType = CryptoObjectType::kUndefined;
    contentProps_.mObjectUid.mGeneratorUid.mQwordLs = 0;
    contentProps_.mObjectUid.mGeneratorUid.mQwordMs = 0;
    contentProps_.mObjectUid.mVersionStamp = 0;
    return netaos::core::Result<void>();
};

netaos::core::Result<KeySlotContentProps> CimplKeySlot::GetContentProps() const noexcept {
    return netaos::core::Result<KeySlotContentProps>::FromValue(contentProps_);
};


netaos::core::Result<cryp::CryptoProvider::Uptr> CimplKeySlot::MyProvider() const noexcept{
    auto uptr = std::make_unique<cryp::CimplCryptoProvider>();
    return netaos::core::Result<cryp::CryptoProvider::Uptr>::FromValue(std::move(uptr));
};

netaos::core::Result<KeySlotPrototypeProps> CimplKeySlot::GetPrototypedProps() const noexcept {
    return netaos::core::Result<KeySlotPrototypeProps>::FromValue(protoProps_);
};

bool CimplKeySlot::IsEmpty() const noexcept {
    #if 0
    CRYP_INFO<<"contentProps_.mAlgId:" <<contentProps_.mAlgId;
    CRYP_INFO<<"contentProps_.mObjectUid.IsNil():"<<contentProps_.mObjectUid.IsNil();
    if((contentProps_.mAlgId == crypto::kAlgIdUndefined) && (contentProps_.mObjectUid.IsNil())) {
        return true;
    }else {
        return false;
    }
    #endif
    return true;
};

netaos::core::Result<IOInterface::Uptr> CimplKeySlot::Open(bool subscribeForUpdates, bool writeable) const noexcept {
    uint64_t IOInterface_ref = 0;
    int32_t ipc_res = CryptoCmClient::Instance().Open(keySlot_ref_.ref, subscribeForUpdates, writeable, IOInterface_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<IOInterface::Uptr>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    CRYP_INFO<<"CreateEncryptorPublicCtx IOInterface_ref.ref:"<<IOInterface_ref;

    CryptoIoContainerRef ioContainerRef;
    ioContainerRef.ref = IOInterface_ref;
    auto uptr = std::make_unique<CimplOInterface>(ioContainerRef);
    return netaos::core::Result<IOInterface::Uptr>::FromValue(std::move(uptr));
};

netaos::core::Result<void> CimplKeySlot::SaveCopy(const IOInterface& container) noexcept {
    int32_t ipc_res = CryptoCmClient::Instance().SaveCopy(keySlot_ref_.ref, dynamic_cast<CimplOInterface*>(const_cast<IOInterface*>(&container))->getContainer().ref);
    // io_container.push_back(&const_cast<IOInterface&>(container));
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<void>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    return netaos::core::Result<void>();
};

}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara