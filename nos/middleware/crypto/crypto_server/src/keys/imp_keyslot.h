#ifndef ARA_CRYPTO_KEYS_IMP_KEY_SLOT_H_
#define ARA_CRYPTO_KEYS_IMP_KEY_SLOT_H_

#include "common/imp_io_interface.h"
#include "cryp/crypto_provider.h"

#include "keys/keyslot.h"
#include "keys/key_slot_prototype_props.h"
#include "keys/key_slot_content_props.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

class ImpKeySlot:public KeySlot {
public:
    using Uptr = std::unique_ptr<ImpKeySlot>;
    netaos::core::Result<void> Clear () noexcept override;
    ImpKeySlot(KeySlotPrototypeProps proProps,KeySlotContentProps conProps):protoProps_(proProps),contentProps_(conProps){}
    ImpKeySlot() = default;
    ~ImpKeySlot () noexcept=default;
    netaos::core::Result<KeySlotContentProps> GetContentProps () const noexcept override;
    netaos::core::Result<cryp::CryptoProvider::Uptr> MyProvider() const noexcept override;
    netaos::core::Result<KeySlotPrototypeProps> GetPrototypedProps() const noexcept override;
    bool IsEmpty() const noexcept override;
    netaos::core::Result<IOInterface::Uptr> Open(bool subscribeForUpdates , bool writeable) const noexcept override;
    netaos::core::Result<void> SaveCopy(const IOInterface& container) noexcept override;
    ImpKeySlot& operator= (const ImpKeySlot &other)=default;
    ImpKeySlot& operator= (ImpKeySlot &&other)=default;
    std::shared_ptr<IOInterface> GetContent() {
        return content_;
    };

    void SetContent(std::shared_ptr<IOInterface> content){
        content_ = content;
    };
private:
    std::shared_ptr<IOInterface> content_;
    KeySlotPrototypeProps protoProps_;
    KeySlotContentProps contentProps_;
};


}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_IMP_KEY_SLOT_H_