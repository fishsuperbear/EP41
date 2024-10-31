#include "cimpl_key_storage_provider.h"

#include "common/crypto_error_domain.h"
#include "cimpl_io_interface.h"
#include "common/crypto_logger.hpp"

#include "keys/elementary_types.h"
#include "cimpl_keyslot.h"
#include "client/crypto_cm_client.h"
// #include "keys/json_parser.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

std::vector<std::tuple<TransactionId, std::uint32_t, std::string>> transaction;

netaos::core::Result<TransactionId> CimplKeyStorageProvider::BeginTransaction(const TransactionScope& targetSlots) noexcept {
    CRYP_INFO<<"BeginTransaction:begin.";
    keys::TransactionId id = 0;
    int32_t ipc_res = CryptoCmClient::Instance().BeginTransaction(targetSlots, id);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<TransactionId>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    return netaos::core::Result<TransactionId>::FromValue(id);
};

netaos::core::Result<void> CimplKeyStorageProvider::CommitTransaction(TransactionId id) noexcept{
    int32_t ipc_res = CryptoCmClient::Instance().CommitTransaction(id);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        return netaos::core::Result<void>::FromError(static_cast<CryptoErrc>(ipc_res));
    }
    return netaos::core::Result<void>();
};

// UpdatesObserver::Uptr CimplKeyStorageProvider::GetRegisteredObserver() const noexcept{
// };

netaos::core::Result<KeySlot::Uptr> CimplKeyStorageProvider::LoadKeySlot(netaos::core::InstanceSpecifier& iSpecify) noexcept{
    auto keyloatid = iSpecify.ToString().data();

    CRYP_INFO<<"LoadKeySlot:begin LoadKeySlot.";
    CryptoSlotRef slot_ref;
    int32_t ipc_res = CryptoCmClient::Instance().LoadKeySlot(keyloatid, slot_ref);
    if (ipc_res != static_cast<int32_t>(CryptoErrc::kSuccess)) {
        CRYP_ERROR << "LoadKeySlot in Server failed. ret:"<<ipc_res;
        return netaos::core::Result<KeySlot::Uptr>::FromError(static_cast<CryptoErrc>(ipc_res));
    }

    std::unique_ptr<CimplKeySlot> uptr(new CimplKeySlot(slot_ref));
    return netaos::core::Result<KeySlot::Uptr>::FromValue(std::move(uptr));
};

// UpdatesObserver::Uptr CimplKeyStorageProvider::RegisterObserver(UpdatesObserver::Uptr observer = nullptr) noexcept{
// };

netaos::core::Result<void> CimplKeyStorageProvider::RollbackTransaction(TransactionId id) noexcept{

    return netaos::core::Result<void>();
};

netaos::core::Result<void> CimplKeyStorageProvider::UnsubscribeObserver(KeySlot& slot) noexcept{

    return netaos::core::Result<void>();
};

}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara