#ifndef ARA_CRYPTO_KEYS_UPDATES_OBSERVER_H_
#define ARA_CRYPTO_KEYS_UPDATES_OBSERVER_H_


namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {
class UpdatesObserver{
public:
    using Uptr = std::unique_ptr<UpdatesObserver>;
    virtual ~UpdatesObserver () noexcept=default;
    virtual void OnUpdate(const TransactionScope& updatedSlots) noexcept = 0;
    UpdatesObserver& operator= (const UpdatesObserver &other)=default;
    UpdatesObserver& operator= (UpdatesObserver &&other)=default;
   private:
   
 
};
}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_UPDATES_OBSERVER_H_