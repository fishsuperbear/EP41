/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file promise.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_PROMISE_HPP_
#define APD_ARA_CORE_PROMISE_HPP_

#include <ara/core/error_code.h>
#include <ara/core/future.h>
#include <ara/core/result.h>
#include <unistd.h>

#include <exception>
#include <future>
#include <mutex>
#include <atomic>

namespace ara {
namespace core {
inline namespace _19_11 {

/**
 * @class Promise
 *
 * @brief ara::core specific variant of std::promise class
 *
 * @tparam T the type of value
 * @tparam E the type of error
 *
 * @uptrace{SWS_CORE_00340}
 */
template <typename T, typename E = ErrorCode>
class Promise {
   public:
    using R = Result<T, E>;

    using Lock = std::unique_lock<std::mutex>;

    /// Alias type for T
    using ValueType = T;

    /**
     * @brief Default constructor.
     *
     * @uptrace{SWS_CORE_00341}
     */
    Promise() { 
        mFuture = std::make_shared<InterFuture<T, E>>( delegate_promise_.get_future() );
    }

    /**
     * @brief Move constructor.
     *
     * @param other the other instance
     *
     * @uptrace{SWS_CORE_00342}
     */
    Promise( Promise &&other ) noexcept
        : delegate_promise_( std::move( other.delegate_promise_ ) ) {
        mFuture = std::move(other.mFuture);
    }

    /**
     * @brief Copy constructor shall be disabled.
     *
     * @uptrace{SWS_CORE_00350}
     */
    Promise( Promise const & ) = delete;

    /**
     * @brief Destructor for Promise objects.
     *
     * @uptrace{SWS_CORE_00349}
     */
    ~Promise() {
        if ( mFuture == nullptr ) {
            return;
        }

        if (0 != mFuture->mCallBack.size()) {
            set_exception(
                std::make_exception_ptr( std::future_error( std::future_errc::broken_promise ) ) );
        }

        mFuture = nullptr;
    };

    /**
     * @brief Move assignment.
     *
     * @param other the other instance
     * @return *this
     *
     * @uptrace{SWS_CORE_00343}
     */
    Promise &operator=( Promise &&other ) noexcept {
        if ( this != &other ) {
            Lock lhsLock( mFuture->mutex_, std::defer_lock );
            Lock rhsLock( other.mFuture->mutex_, std::defer_lock );
            std::lock( lhsLock, rhsLock );

            delegate_promise_ = std::move( other.delegate_promise_);
            mFuture = std::move( other.mFuture );
        }
        return *this;
    }

    /**
     * @brief Copy assignment operator shall be disabled.
     *
     * @uptrace{SWS_CORE_00351}
     */
    Promise &operator=( Promise const & ) = delete;

    /**
     * @brief Swap the contents of this instance with another one’s.
     *
     * @param other the other instance
     *
     * @uptrace{SWS_CORE_00352}
     */
    void swap( Promise &other ) noexcept {
        Lock lhsLock( mFuture->mutex_, std::defer_lock );
        Lock rhsLock( other.mFuture->mutex_, std::defer_lock );
        std::lock( lhsLock, rhsLock );

        using std::swap;
        swap(delegate_promise_, other.delegate_promise_);
        swap(mFuture, other.mFuture);
    }

    /**
     * @brief Return the associated Future.
     * The returned Future is set as soon as this Promise receives the result or an error. This
     * method must only be called once as it is not allowed to have multiple Futures per Promise.
     *
     * @return a Future
     *
     * @uptrace{SWS_CORE_00344}
     */
    Future<T, E> get_future() { return Future<T, E>( mFuture ); }

    /**
     * @brief Copy a value into the shared state and make the state ready.
     *
     * @param value the value to store
     *
     * @uptrace{SWS_CORE_00345}
     */
    void set_value( T const &value ) {
        R    r = R::FromValue( std::move( value ) );
        delegate_promise_.set_value( r );

        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

    /**
     * @brief Move a value into the shared state and make the state ready.
     *
     * @param value the value to store
     *
     * @uptrace{SWS_CORE_00346}
     */
    void set_value( T &&value ) {
        R    r = R::FromValue( value );
        delegate_promise_.set_value( r );
        
        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

    /**
     * @brief Move an error into the shared state and make the state ready
     *
     * @param err the error to store
     *
     * @uptrace{SWS_CORE_00353}
     */
    void SetError( E &&err ) {
        R r = R::FromError( std::move( err ) );
        delegate_promise_.set_value( r );

        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        uint32_t size = list.size();
        callCBFunc(list);
    }

    /**
     * @brief Copy an error into the shared state and make the state ready.
     *
     * @param err the error to store
     *
     * @uptrace{SWS_CORE_00354}
     */
    void SetError( E const &err ) {
        R r = R::FromError( err );
        delegate_promise_.set_value( r );

        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

    /**
     * @brief Sets an exception.
     *
     * Calling Get() on the associated Future will rethrow the exception in the
     * context the Future's method was
     * called in.
     *
     * @param p exception_ptr to set
     *
     * @note This method is DEPRECATED. The exception is defined by the error code.
     */
    void set_exception( std::exception_ptr p ) {
        delegate_promise_.set_exception( p );

        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

   private:
    void callCBFunc(std::vector<std::function<void( void )>>& list) {
        uint32_t size = list.size();
        for ( unsigned int i = 0U; i < size; i++ ) {
            list[ i ]();
        }
        list.clear();
    }
    std::promise<R> delegate_promise_;
    std::shared_ptr<InterFuture<T, E>> mFuture;
};

/**
 * @brief Explicit specialization of Promise for void
 *
 * @uptrace{SWS_CORE_06340}
 */
template <typename E>
class Promise<void, E> final {
   public:
    using R = Result<void, E>;

    using Lock = std::unique_lock<std::mutex>;

    /// @copydoc Promise::ValueType
    using ValueType = void;

    /**
     * @brief Default constructor.
     *
     * @uptrace{SWS_CORE_06341}
     */
    Promise() {
        mFuture = std::make_shared<InterFuture<void, E>>( delegate_promise_.get_future() );
    }

    /**
     * @brief Move constructor.
     *
     * @param other the other instance
     *
     * @uptrace{SWS_CORE_06342}
     */
    Promise( Promise &&other ) noexcept
        : delegate_promise_( std::move( other.delegate_promise_ ) ) {
        mFuture = std::move(other.mFuture);
    }

    /**
     * @brief Copy constructor shall be disabled.
     *
     * @uptrace{SWS_CORE_06350}
     */
    Promise( Promise const & ) = delete;

    /**
     * @brief Destructor for Promise objects.
     *
     * @uptrace{SWS_CORE_06349}
     */
    ~Promise() {
        if ( mFuture == nullptr ) {
            return;
        }

        if (0 != mFuture->mCallBack.size()) {
            set_exception(
                std::make_exception_ptr( std::future_error( std::future_errc::broken_promise ) ) );
        }

        mFuture = nullptr;
    };

    /**
     * @brief Move assignment.
     *
     * @param other the other instance
     * @return *this
     *
     * @uptrace{SWS_CORE_06343}
     */
    Promise &operator=( Promise &&other ) noexcept {
        if ( this != &other ) {
            Lock lhsLock( mFuture->mutex_, std::defer_lock );
            Lock rhsLock( other.mFuture->mutex_, std::defer_lock );
            std::lock( lhsLock, rhsLock );

            delegate_promise_ = std::move( other.delegate_promise_);
            mFuture = std::move( other.mFuture );
        }
        return *this;
    }

    /**
     * @brief Copy assignment operator shall be disabled.
     *
     * @uptrace{SWS_CORE_06351}
     */
    Promise &operator=( Promise const & ) = delete;

    /**
     * @brief Swap the contents of this instance with another one’s.
     *
     * @param other the other instance
     *
     * @uptrace{SWS_CORE_06352}
     */
    void swap( Promise &other ) noexcept {
        Lock lhsLock( mFuture->mutex_, std::defer_lock );
        Lock rhsLock( other.mFuture->mutex_, std::defer_lock );
        std::lock( lhsLock, rhsLock );

        using std::swap;
        swap( delegate_promise_, other.delegate_promise_ );
        swap( mFuture, other.mFuture );
    }

    /**
     * @brief Return the associated Future.
     *
     * @return a Future
     *
     * @uptrace{SWS_CORE_06344}
     */
    Future<void, E> get_future() { return Future<void, E>( mFuture ); }

    /**
     * @brief Make the shared state ready.
     *
     * @uptrace{SWS_CORE_06345}
     */
    void set_value() {
        delegate_promise_.set_value( R::FromValue() );

        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

    /**
     * @brief Move an error into the shared state and make the state ready.
     *
     * @param err the error to store
     *
     * @uptrace{SWS_CORE_06353}
     */
    void SetError( E &&err ) {
        R r = R::FromError( err );
        delegate_promise_.set_value( r );

        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

    /**
     * @brief Copy an error into the shared state and make the state ready.
     *
     * @param err the error to store
     *
     * @uptrace{SWS_CORE_06354}
     */
    void SetError( E const &err ) {
        R r = R::FromError( std::move( err ) );
        delegate_promise_.set_value( r );

        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

    /**
     * @brief Sets an exception.
     *
     * Calling Get() on the associated Future will rethrow the exception in the
     * context the Future's method was
     * called in.
     *
     * @param p exception_ptr to set
     *
     * @note This method is DEPRECATED. The exception is defined by the error code.
     */
    void set_exception( std::exception_ptr p ) {
        delegate_promise_.set_exception( p );
        
        Lock lock( mFuture->mutex_ );
        std::vector<std::function<void( void )>> list = std::move(mFuture->mCallBack);
        lock.unlock();
        callCBFunc(list);
    }

   private:
    void callCBFunc(std::vector<std::function<void( void )>>& list) {
        uint32_t size = list.size();
        for ( unsigned int i = 0U; i < size; i++ ) {
            list[ i ]();
        }
    }
    std::promise<R> delegate_promise_;
    std::shared_ptr<InterFuture<void, E>> mFuture;
};
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_PROMISE_HPP_
/* EOF */
