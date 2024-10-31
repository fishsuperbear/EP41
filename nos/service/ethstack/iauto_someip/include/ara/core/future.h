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
 * @file future.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_FUTURE_HPP_
#define APD_ARA_CORE_FUTURE_HPP_

#include <ara/core/error_code.h>
#include <ara/core/exception.h>
#include <ara/core/future_error_domain.h>
#include <ara/core/result.h>

#include <cassert>
#include <chrono>
#include <future>
#include <vector>

namespace ara {
namespace core {
inline namespace _19_11 {

/* Forward declaration */
template <typename, typename>
class Promise;

/**
 * @brief Specifies the state of a Future as returned by wait_for() and
 * wait_until().
 *
 * These definitions are equivalent to the ones from std::future_status.
 * However, the
 * item std::future_status::deferred is not supported here.
 *
 * @uptrace{SWS_CORE_00361}
 */
enum class future_status : uint8_t {
    ready = 0, /**< the shared state is ready */
    timeout,   /**< the shared state did not become ready before the specified timeout has passed */
    intererror /** custom error: there has a error in ara::core::future */
};

/**
 * @brief print the errorCode and the explanatory string.
 *
 * @param out [in] std::ostream object
 * @param ex [in] the exception
 * @return std::ostream object
 */
inline std::ostream &operator<<( std::ostream &out, FutureException const &ex ) {
    return ( out << "FutureException: " << ex.Error() << " (" << ex.what() << ")" );
}

template <typename T, typename E = ErrorCode>
struct InterFuture {
    using R = Result<T, E>;

    std::mutex                               mutex_;
    std::future<R>                           delegate_future_;
    std::vector<std::function<void( void )>> mCallBack;

    InterFuture( std::future<R> &&future_ ) : delegate_future_( std::move( future_ ) ) {}

    ~InterFuture() { mCallBack.clear(); }
};

/**
 * @brief Provides ara::core specific Future operations to collect the results
 * of an asynchronous call.
 *
 * Much of its functionality is delegated to std::future and all methods that
 * resemble std::future are guaranteed to
 * behave the same.
 *
 * If the valid() member function of an instance returns true, all other methods
 * are guaranteed to work on that
 * instance. Otherwise, they may fail with or without an exception. A thrown
 * exception will be of type
 * std::future_error.
 *
 * Having an invalid future will usually happen when the future was moved from
 * using the move constructor or move
 * assignment.
 *
 * @tparam T  the type of values
 * @tparam E  the type of errors
 *
 * @uptrace{SWS_CORE_00321} {SWS_CORE_06221}
 */
template <typename T, typename E = ErrorCode>
class Future final {

   private:
    template<typename A>
    struct _is_then_func : public std::true_type {};

   public:
    using R = Result<T, E>;

    using Lock = std::unique_lock<std::mutex>;

    /// Alias type for T
    using ValueType = T;
    /// Alias type for the Promise type collaborating with this Future type
    using PromiseType = Promise<T, E>;

    /**
     * @brief Default constructor
     *
     * @uptrace{SWS_CORE_00322} {SWS_CORE_06222}
     */
    Future() noexcept = default;

    /**
     * @uptrace{SWS_CORE_00334} {SWS_CORE_06234}
     */
    Future( Future const & ) = delete;

    /**
     * @brief Move construct from another instance.
     *
     * @param other  the other instance
     *
     * @uptrace{SWS_CORE_00323} {SWS_CORE_06223}
     */
    Future( Future &&other ) noexcept : mStruct( std::move( other.mStruct ) ) {}

    /**
     * @brief Destructor for Future objects
     *
     * This will also disable any callback that has been set.
     *
     * @uptrace{SWS_CORE_00333} {SWS_CORE_06233}
     */
    ~Future() {
        if ( mStruct != nullptr ) {
            mStruct = nullptr;
        }
    };

    // @uptrace{SWS_CORE_00335} {SWS_CORE_06235}
    Future &operator=( Future const & ) = delete;

    /**
     * @brief Move assign from another instance.
     *
     * @param other the other instance
     * @return *this
     *
     * @uptrace{SWS_CORE_00325} {SWS_CORE_06225}
     */
    Future &operator=( Future &&other ) noexcept {
        if ( this != &other ) {
            std::shared_ptr<InterFuture<T, E>> tmpStruct = nullptr;
            if ( other.mStruct == nullptr ) {
                if ( this->mStruct == nullptr ) {
                    return *this;
                } else {
                    tmpStruct = this->mStruct;
                    Lock lock( tmpStruct->mutex_ );
                    this->mStruct = nullptr;
                    tmpStruct     = nullptr;
                }
                return *this;
            }

            if ( mStruct == nullptr ) {
                Lock lock( other.mStruct->mutex_ );
                mStruct = std::move( other.mStruct );
                return *this;
            } else {
                tmpStruct = this->mStruct;
                Lock lhsLock( tmpStruct->mutex_, std::defer_lock );
                Lock rhsLock( other.mStruct->mutex_, std::defer_lock );
                std::lock( lhsLock, rhsLock );
                mStruct   = std::move( other.mStruct );
                tmpStruct = nullptr;
            }
        }
        return *this;
    }

    /**
     * @brief Get the result.
     *
     * @return a Result with either a value or an error
     *
     * @uptrace{SWS_CORE_00336} {SWS_CORE_06236}
     */
    R GetResult() noexcept {

        if ( mStruct == nullptr ) {
            future_errc err = future_errc::unknow_error;
            return R::FromError( std::move( err ) );
        }

        try {
            return mStruct->delegate_future_.get();
        } catch ( std::future_error const &ex ) {
            std::error_code const &ec = ex.code();
            future_errc            err;
            if ( std::future_errc::broken_promise == ec ) {
                err = future_errc::broken_promise;
            } else if ( std::future_errc::future_already_retrieved == ec ) {
                err = future_errc::future_already_retrieved;
            } else if ( std::future_errc::promise_already_satisfied == ec ) {
                err = future_errc::promise_already_satisfied;
            } else if ( std::future_errc::no_state == ec ) {
                err = future_errc::no_state;
            } else {
                // Should rather use a vendor/demonstrator-specific ErrorDomain here?
                err = future_errc::unknow_error;
            }
            return R::FromError( std::move( err ) );
        } catch ( ara::core::Exception const &e ) {
            return R::FromError( std::move( e.Error() ) );
        }
    }

    /**
     * @brief Get the value.
     *
     * This call blocks until the result or an exception is available.
     *
     * @returns value of type T
     *
     * @uptrace{SWS_CORE_00326} {SWS_CORE_06226}
     */
    T get() { return GetResult().ValueOrThrow(); }

    /**
     * @brief Checks if the future is valid, i.e. if it has a shared state.
     *
     * @returns true if the future is usable, false otherwise
     *
     * @uptrace{SWS_CORE_00327} {SWS_CORE_06227}
     */
    bool valid() const noexcept {
        if ( mStruct == nullptr ) {
            return false;
        }
        return mStruct->delegate_future_.valid();
    }

    /**
     * @brief Waits for a value or an exception to be available.
     *
     * After this method returns, get() is guaranteed to not block and is_ready()
     * will return true.
     *
     * @uptrace{SWS_CORE_00328} {SWS_CORE_06228}
     */
    void wait() const {
        if ( mStruct == nullptr ) {
            std::cerr << "future::wait mStruct is nullptr!" << std::endl;
            return;
        }
        mStruct->delegate_future_.wait();
    }

    /**
     * @brief Wait for the given period, or until a value or an error is available.
     *
     * If the Future becomes ready or the timeout is reached, the method returns.
     *
     * @tparam Rep  representation type used as the type for the internal count object.
     * @tparam Period  the ratio type that represents a period in seconds.
     * @param timeout_duration maximal duration to wait for
     * @returns status that indicates whether the timeout hit or if a value is
     * available
     *
     * @uptrace{SWS_CORE_00329} {SWS_CORE_06229}
     */
    template <typename Rep, typename Period>
    future_status wait_for( std::chrono::duration<Rep, Period> const &timeout_duration ) const {

        if ( mStruct == nullptr ) {
            std::cerr << "this std::future_status should not occur in our setup" << std::endl;
            return future_status::intererror;
        }

        switch ( mStruct->delegate_future_.wait_for( timeout_duration ) ) {
            case std::future_status::ready:
                return future_status::ready;
            case std::future_status::timeout:
                return future_status::timeout;
            default:
                std::cerr << "this std::future_status should not occur in our setup" << std::endl;
                break;
        }
        return future_status::intererror;
    }

    /**
     * @brief Wait until the given time, or until a value or an error is available.
     *
     * If the Future becomes ready or the time is reached, the method returns.
     *
     * @tparam Clock  the clock class such as system_clock,steady_clock
     * @tparam Duration  the duration type to represetnt the time point
     * @param deadline latest point in time to wait
     * @returns status that indicates whether the time was reached or if a value is available
     *
     * @uptrace{SWS_CORE_00330} {SWS_CORE_06230}
     */
    template <typename Clock, typename Duration>
    future_status wait_until( std::chrono::time_point<Clock, Duration> const &deadline ) const {

        if ( mStruct == nullptr ) {
            std::cerr << "this std::future_status should not occur in our setup" << std::endl;
            return future_status::intererror;
        }

        switch ( mStruct->delegate_future_.wait_until( deadline ) ) {
            case std::future_status::ready:
                return future_status::ready;
            case std::future_status::timeout:
                return future_status::timeout;
            default:
                std::cerr << "this std::future_status should not occur in our setup" << std::endl;
        }
        return future_status::intererror;
    }

    /**
     * @brief Register a function that gets called when the future becomes ready.
     *
     * When @a func is called, it is guaranteed that get() will not block.
     *
     * @a func may be called in the context of this call or in the context of
     * Promise::set_value()
     * or Promise::set_exception() or somewhere else.
     *
     * @param func a callable to register to get the Future result or an exception
     * @return a new Future instance for the result of the continuation
     *
     * @uptrace{SWS_CORE_00331} {SWS_CORE_06231}
     */
    template<typename F>
    std::enable_if_t<_is_then_func<std::result_of_t<std::decay_t<F>()>>::value, Future<T>&> then(F&& func) {
        if ( is_ready() ) {
            func();
        }
        else {
            Lock lock( mStruct->mutex_ );
            if ( is_ready()) {
                lock.unlock();
                func();
            }
            else {
                mStruct->mCallBack.push_back( func );
            }
        }
        return *this;
    }

    template<typename F>
    std::enable_if_t<_is_then_func<std::result_of_t<std::decay_t<F>(Future<T>&&)>>::value, Future<T>&> then(F&& func) {
        if ( is_ready() ) {
            func(Future<T>(mStruct));
        }
        else {
            Lock lock( mStruct->mutex_ );
            if ( is_ready()) {
                lock.unlock();
                func(Future<T>(mStruct));
            }
            else {
                std::weak_ptr<InterFuture<T, E>> wp = mStruct;
                std::function<void(Future<T>&&)> tmp = std::move(func);
                mStruct->mCallBack.push_back( [wp, tmp](){
                    if ( auto sp_ptr = wp.lock()) {
                        tmp(Future<T>(sp_ptr));
                    }
                    else {
                        std::cerr << "future::then weak ptr mStruct is null!" << std::endl;
                    }
                    
                } );
            }
        }
        return *this;
    }

    /**
     * @brief Return whether the asynchronous operation has finished.
     * True when the future contains either a result or an exception.
     *
     * If is_ready() returns true, get() and the wait calls are guaranteed to not
     * block.
     *
     * @returns true if the Future contains data, false otherwise
     *
     * @uptrace{SWS_CORE_00332} {SWS_CORE_06232}
     */
    bool is_ready() const {
        if ( mStruct == nullptr ) {
            std::cerr << "future::is_ready mStruct is nullptr!" << std::endl;
            return false;
        }
        return std::future_status::ready ==
               mStruct->delegate_future_.wait_for( std::chrono::seconds::zero() );
    }

   private:
    /**
     * @brief Constructs a Future from a given std::future and a pointer to the
     * callBack's vector.
     *
     * @param delegate_future std::future instance
     * @param callBack callBack's vector
     */
    Future( std::shared_ptr<InterFuture<T, E>> sInterFuture ) : mStruct( sInterFuture ) {}

    std::shared_ptr<InterFuture<T, E>> mStruct;

    template <typename, typename>
    friend class Promise;
};
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_FUTURE_HPP_
/* EOF */
