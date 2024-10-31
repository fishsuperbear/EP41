#ifndef INCLUDE_COM_SAMPLE_PTR_H_
#define INCLUDE_COM_SAMPLE_PTR_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <memory>
#include "ara/com/e2e/e2e_types.h"

namespace ara {
namespace com {
inline namespace _19_11 {

/**
 * @brief Pointer to a sample
 *
 * The semantics of a SamplePtr are the same as for an std::shared_ptr<T>.
 *
 * @note This implementation might be changed by product vendor.
 *
 * @uptrace{SWS_CM_00306}
 */
template <typename T>
class SamplePtr {
    using  ProfileCheckStatus = ara::com::e2e::ProfileCheckStatus;

   public:
    /* constexpr shared_ptr() noexcept */
    explicit constexpr SamplePtr( ProfileCheckStatus profileCheckState = ProfileCheckStatus::kOk ) noexcept
        : dataPtr_(), profileCheckState_( profileCheckState ) {}

    /* template<class Y>shared_ptr(Y* ptr) ctor can throw */
    explicit SamplePtr( T* ptr, ProfileCheckStatus profileCheckState = ProfileCheckStatus::kOk )
        : dataPtr_( ptr ), profileCheckState_( profileCheckState ) {}

    /* template<class Y>shared_ptr(Y* ptr) ctor can throw */
    explicit SamplePtr( std::shared_ptr<T> ptr, ProfileCheckStatus profileCheckState = ProfileCheckStatus::kOk )
        : dataPtr_( ptr ), profileCheckState_( profileCheckState ) {}

    /* shared_ptr( const shared_ptr& r ) noexcept; */
    SamplePtr( const SamplePtr<T>& r ) noexcept
        : dataPtr_( r.dataPtr_ ), profileCheckState_( r.profileCheckState_ ) {}

    /* shared_ptr( const shared_ptr&& r ) noexcept; */
    SamplePtr( SamplePtr<T>&& r ) noexcept
        : dataPtr_( std::move( r.dataPtr_ ) ), profileCheckState_( std::move( r.profileCheckState_ ) ) {}

    /* shared_ptr& operator=( const shared_ptr& r ) noexcept; */
    SamplePtr& operator=( const SamplePtr<T>& r ) noexcept {
        dataPtr_           = r.dataPtr_;
        profileCheckState_ = r.profileCheckState_;
        return *this;
    }

    /* shared_ptr& operator=( shared_ptr&& r ) noexcept; */
    SamplePtr& operator=( SamplePtr<T>&& r ) noexcept {
        dataPtr_           = std::move( r.dataPtr_ );
        profileCheckState_ = std::move( r.profileCheckState_ );
        return *this;
    }

    /* T& shared_ptr::operator*() const noexcept; */
    T& operator*() const noexcept { return dataPtr_.operator*(); }

    /* T* shared_ptr::operator->() const noexcept; */
    T* operator->() const noexcept { return dataPtr_.operator->(); }

    /* @uptrace{SWS_CM_90420}*/
    ProfileCheckStatus  GetProfileCheckStatus() const noexcept { return profileCheckState_; }

   private:
    std::shared_ptr<T> dataPtr_;
    ProfileCheckStatus profileCheckState_;
};

}  // inline namespace _19_11
}  // namespace com
}  // namespace ara

#endif  // INCLUDE_COM_SAMPLE_PTR_H_
/* EOF */
