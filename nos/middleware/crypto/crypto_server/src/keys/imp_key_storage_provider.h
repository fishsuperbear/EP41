#ifndef ARA_CRYPTO_KEYS_IMP_KEY_STORAGE_PROVIDER_H_
#define ARA_CRYPTO_KEYS_IMP_KEY_STORAGE_PROVIDER_H_

#include <memory>
#include "core/instance_specifier.h"
#include "keys/key_storage_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {
class IMPKeyStorageProvider:public KeyStorageProvider{
public:
    using Uptr = std::unique_ptr<IMPKeyStorageProvider>;
    netaos::core::Result<TransactionId> BeginTransaction(const TransactionScope& targetSlots) noexcept override;
    netaos::core::Result<void> CommitTransaction(TransactionId id) noexcept override;
    ~IMPKeyStorageProvider() noexcept = default;
    // UpdatesObserver::Uptr GetRegisteredObserver() const noexcept override;
    netaos::core::Result<KeySlot::Uptr> LoadKeySlot(netaos::core::InstanceSpecifier& iSpecify) noexcept override;
    // UpdatesObserver::Uptr RegisterObserver(UpdatesObserver::Uptr observer = nullptr) noexcept override;
    netaos::core::Result<void> RollbackTransaction(TransactionId id) noexcept override;
    netaos::core::Result<void> UnsubscribeObserver(KeySlot& slot) noexcept override;
    IMPKeyStorageProvider& operator=(const IMPKeyStorageProvider& other) = default;
    IMPKeyStorageProvider& operator= (IMPKeyStorageProvider &&other) = default;
private:
   
 
};
}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_IMP_KEY_STORAGE_PROVIDER_H_