#ifndef ARA_CRYPTO_KEYS_IMP_KEY_SLOT_H_
#define ARA_CRYPTO_KEYS_IMP_KEY_SLOT_H_

#include "cimpl_io_interface.h"
#include "cryp/crypto_provider.h"

#include "keys/keyslot.h"
#include "keys/key_slot_prototype_props.h"
#include "keys/key_slot_content_props.h"
#include "common/inner_types.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

class CimplKeySlot:public KeySlot {
public:
    using Uptr = std::unique_ptr<CimplKeySlot>;
    netaos::core::Result<void> Clear () noexcept override;
    CimplKeySlot(KeySlotPrototypeProps proProps,KeySlotContentProps conProps):protoProps_(proProps),contentProps_(conProps){}
    CimplKeySlot(const CryptoSlotRef& keySlot_ref);
    CimplKeySlot() = default;
    ~CimplKeySlot();
    netaos::core::Result<KeySlotContentProps> GetContentProps () const noexcept override;
    netaos::core::Result<cryp::CryptoProvider::Uptr> MyProvider() const noexcept override;
    netaos::core::Result<KeySlotPrototypeProps> GetPrototypedProps() const noexcept override;
    bool IsEmpty() const noexcept override;
    netaos::core::Result<IOInterface::Uptr> Open(bool subscribeForUpdates , bool writeable) const noexcept override;
    netaos::core::Result<void> SaveCopy(const IOInterface& container) noexcept override;
    CimplKeySlot& operator= (const CimplKeySlot &other)=default;
    CimplKeySlot& operator= (CimplKeySlot &&other)=default;
    std::vector<std::uint8_t> GetContent(){
        return content_;
    };

    CryptoSlotRef getSlotRef() {
        return keySlot_ref_;
    }

    // std::vector<cryp::IOInterface*> io_container;

private:
    std::vector<std::uint8_t> content_;
    KeySlotPrototypeProps protoProps_;
    KeySlotContentProps contentProps_;

    mutable CryptoSlotRef keySlot_ref_;
};


}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_IMP_KEY_SLOT_H_