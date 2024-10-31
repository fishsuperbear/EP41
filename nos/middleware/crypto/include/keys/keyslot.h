#ifndef ARA_CRYPTO_KEYS_KEY_SLOT_H_
#define ARA_CRYPTO_KEYS_KEY_SLOT_H_

#include "core/result.h"
#include "keys/key_slot_prototype_props.h"
#include "keys/key_slot_content_props.h"
#include "cryp/crypto_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

// using namespace netaos::crypto::cryp;
class KeySlot{
public:
    using Uptr = std::unique_ptr<KeySlot>;
    virtual netaos::core::Result<void> Clear () noexcept=0;
    virtual ~KeySlot () noexcept=default;
    virtual netaos::core::Result<KeySlotContentProps> GetContentProps () const noexcept=0;
    virtual netaos::core::Result<cryp::CryptoProvider::Uptr> MyProvider() const noexcept = 0;
    virtual netaos::core::Result<KeySlotPrototypeProps> GetPrototypedProps() const noexcept = 0;
    virtual bool IsEmpty() const noexcept = 0;
    virtual netaos::core::Result<IOInterface::Uptr> Open(bool subscribeForUpdates = false, bool writeable = false) const noexcept = 0;

    virtual netaos::core::Result<void> SaveCopy(const IOInterface& container) noexcept = 0;
    KeySlot& operator= (const KeySlot &other)=default;
    KeySlot& operator= (KeySlot &&other)=default;
private:
   
};

}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_KEY_SLOT_H_