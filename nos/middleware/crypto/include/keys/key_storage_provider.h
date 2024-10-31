#ifndef ARA_CRYPTO_KEYS_KEY_STORAGE_PROVIDER_H_
#define ARA_CRYPTO_KEYS_KEY_STORAGE_PROVIDER_H_
#include <memory>
#include "core/result.h"
#include "core/instance_specifier.h"
#include "keyslot.h"
#include "elementary_types.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {
class KeyStorageProvider{
public:
    using Uptr = std::unique_ptr<KeyStorageProvider>;
    virtual netaos::core::Result<TransactionId> BeginTransaction(const TransactionScope& targetSlots) noexcept = 0;
    virtual netaos::core::Result<void> CommitTransaction(TransactionId id) noexcept = 0;
    virtual ~KeyStorageProvider() noexcept = default;
    // virtual UpdatesObserver::Uptr GetRegisteredObserver() const noexcept = 0;
    virtual netaos::core::Result<KeySlot::Uptr> LoadKeySlot (netaos::core::InstanceSpecifier& iSpecify) noexcept=0;
    // virtual netaos::core::Result<KeySlot::Uptr> LoadKeySlot(std::string uuid) noexcept = 0;
    // virtual UpdatesObserver::Uptr RegisterObserver(UpdatesObserver::Uptr observer = nullptr) noexcept = 0;
    virtual netaos::core::Result<void> RollbackTransaction(TransactionId id) noexcept = 0;
    virtual netaos::core::Result<void> UnsubscribeObserver(KeySlot& slot) noexcept = 0;
    // KeyStorageProvider& operator=(const KeyStorageProvider& other) = default;
    // KeyStorageProvider& operator= (KeyStorageProvider &&other)=default;
   private:
   
 
};
}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_KEY_STORAGE_PROVIDER_H_