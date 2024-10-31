/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: handle DDSDriver to service find and sub pub
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_PROLOC_MEMORY_FACTORY_H
#define VRTF_VCC_PROLOC_MEMORY_FACTORY_H
#include <mutex>
#include <set>
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/utils/thread_pool.h"
#include "vrtf/driver/proloc/proloc_driver_types.h"
#include "vrtf/driver/proloc/proloc_memory_manager.h"
#include "vrtf/vcc/api/raw_buffer.h"
#include "vrtf/vcc/api/recv_buffer.h"
namespace vrtf {
namespace driver {
namespace proloc {
using RawBufferType = std::pair<std::vector<std::uint8_t>, size_t>;
template<class... Args>
class ProlocRequestMemory : public ProlocMethodManager {
public:
    template<std::size_t I, class... AArgs>
    class Parameters;

    template<std::size_t I, class Head, class... PArgs>
    class Parameters<I, Head, PArgs...> {
    public:
        using Type = typename Parameters<I - 1, PArgs...>::Type;
    };

    template<class Head, class... PArgs>
    class Parameters<0, Head, PArgs...> {
    public:
        using Type = Head;
    };

    template<class Head>
    class Parameters<0, Head> {
    public:
        using Type = Head;
    };

    /// \brief ArgType<I> is used to get the Ith parameter's type
    template<std::size_t I>
    using ArgType = typename Parameters<I, Args...>::Type;
    using RequestData = std::map<const std::uint8_t*, std::shared_ptr<std::tuple<Args...>>>;
    using ClientMap = std::map<ProlocEntityIndex, RequestData>;
    ProlocRequestMemory()
        : ProlocMethodManager(), methodData_(), clientRequestMutex_(),
          logInstance_(ara::godel::common::log::Log::GetLog("CM")) {}
    ~ProlocRequestMemory() override {}
    ProlocRequestMemory(const ProlocRequestMemory& other) = default;
    ProlocRequestMemory& operator=(ProlocRequestMemory const &prolocEventMemory) = default;

    std::uint8_t* AllocateRequestBuffer(Args const& ...args, ProlocEntityIndex const& index)
    {
        std::lock_guard<std::mutex> const guard {clientRequestMutex_};
        typename ClientMap::iterator const iter {methodData_.find(index)};
        std::shared_ptr<std::tuple<Args...>> tuplePtr {std::make_shared<std::tuple<Args...>>(std::make_tuple(args...))};
        std::uint8_t* data {reinterpret_cast<std::uint8_t*>(tuplePtr.get())};
        if (iter == methodData_.end()) {
            RequestData requestData;
            static_cast<void>(requestData.emplace(data, tuplePtr));
            static_cast<void>(methodData_.emplace(index, requestData));
            return data;
        }
        static_cast<void>(iter->second.emplace(data, tuplePtr));
        return data;
    }

    void UnRegisterMethod(const ProlocEntityIndex &index) override
    {
        std::lock_guard<std::mutex> const guard {clientRequestMutex_};
        typename ClientMap::const_iterator const iter {methodData_.find(index)};
        if (iter != methodData_.end()) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->debug() << "Proloc unregister method server index [" << index.GetProlocInfo();
            static_cast<void>(methodData_.erase(iter));
        }
    }

    void ReturnLoan(const std::uint8_t* data, const ProlocEntityIndex &index) override
    {
        std::lock_guard<std::mutex> const guard {clientRequestMutex_};
        typename ClientMap::iterator const iter {methodData_.find(index)};
        if (iter == methodData_.end()) {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->warn() << "Proloc request return loan cannot find index [" << index.GetProlocInfo() << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return;
        }
        typename RequestData::const_iterator const dataIter = iter->second.find(data);
        if (dataIter != iter->second.cend()) {
            static_cast<void>(iter->second.erase(dataIter));
        } else {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->warn() << "Proloc request return loan unknon address[" << index.GetProlocInfo() << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
    }

    template <std::size_t I>
    typename std::decay<ArgType<I>>::type GetRequestValue(
        const std::uint8_t* data, const ProlocEntityIndex &index) noexcept
    {
        std::lock_guard<std::mutex> const guard {clientRequestMutex_};
        typename ClientMap::const_iterator iter {methodData_.find(index)};
        if (iter == methodData_.end()) {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->warn() << "Proloc get request value cannot find index [" << index.GetProlocInfo() << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return typename std::decay<ArgType<I>>::type {};
        }
        auto const dataIter = iter->second.find(data);
        if (dataIter != iter->second.end()) {
            std::tuple<Args...> &tuple = *(dataIter->second);
            return std::get<I>(tuple);
        }
        /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
        /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
        logInstance_->warn() << "Proloc get unknow request value [" << index.GetProlocInfo() << "]";
        /* AXIVION enable style AutosarC++19_03-A5.0.1 */
        /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        return typename std::decay<ArgType<I>>::type {};
    }

private:
    ClientMap methodData_;
    std::mutex clientRequestMutex_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
};


template<typename ReplyType>
class ProlocReplyMemory : public ProlocMethodManager {
public:
    using ReplyData = std::map<const std::uint8_t*, std::shared_ptr<vrtf::core::Result<ReplyType>>>;
    using ClientMap = std::map<ProlocEntityIndex, ReplyData>;
    ProlocReplyMemory()
        : ProlocMethodManager(), methodData_(), clientReplyMutex_(),
          logInstance_(ara::godel::common::log::Log::GetLog("CM")) {}
    ~ProlocReplyMemory() override {}
    ProlocReplyMemory(const ProlocReplyMemory& other) = default;
    ProlocReplyMemory& operator=(ProlocReplyMemory const &prolocEventMemory) = default;

    void ReturnLoan(const std::uint8_t* data, const ProlocEntityIndex &index) override
    {
        std::lock_guard<std::mutex> const guard {clientReplyMutex_};
        typename ClientMap::iterator const iter {methodData_.find(index)};
        if (iter == methodData_.end()) {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->warn() << "Proloc reply return loan cannot find index [" << index.GetProlocInfo() << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return;
        }
        typename ReplyData::const_iterator dataIter = iter->second.find(data);
        if (dataIter != iter->second.end()) {
            static_cast<void>(iter->second.erase(dataIter));
        } else {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->warn() << "Proloc reply return loan unknon address[" << index.GetProlocInfo() << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        }
    }

    void UnRegisterMethod(const ProlocEntityIndex &index) override
    {
        std::lock_guard<std::mutex> const guard {clientReplyMutex_};
        typename ClientMap::const_iterator iter {methodData_.find(index)};
        if (iter != methodData_.end()) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->debug() << "Proloc unregister method server index [" << index.GetProlocInfo();
            static_cast<void>(methodData_.erase(iter));
        }
    }

    std::uint8_t* AllocateReplyBuffer(vrtf::core::Result<ReplyType>& data, const ProlocEntityIndex &index,
        const std::shared_ptr<vrtf::vcc::api::types::MethodMsg> &msg)
    {
        std::lock_guard<std::mutex> const guard {clientReplyMutex_};
        std::uint8_t* returnData {nullptr};
        if (data.HasValue()) {
            msg->SetMsgType(false);
        } else {
            auto error = data.Error();
            if (error.Domain() == ara::core::GetFutureErrorDomain()) {
                ara::core::ThrowOrTerminate<vrtf::core::FutureException>(error);
                return returnData;
            }
            msg->SetMsgType(true);
        }
        std::shared_ptr<vrtf::core::Result<ReplyType>> result {
            std::make_shared<vrtf::core::Result<ReplyType>>(data)};
        returnData = reinterpret_cast<std::uint8_t*>(result.get());
        typename ClientMap::iterator iter {methodData_.find(index)};
        if (iter == methodData_.end()) {
            ReplyData replyData;
            static_cast<void>(replyData.emplace(returnData, result));
            static_cast<void>(methodData_.emplace(index, replyData));
            return returnData;
        }
        static_cast<void>(iter->second.emplace(returnData, result));
        return returnData;
    }

    std::shared_ptr<vrtf::core::Result<ReplyType>> GetReplyResult(
        const std::uint8_t *data, const ProlocEntityIndex &index)
    {
        std::lock_guard<std::mutex> const guard {clientReplyMutex_};
        typename ClientMap::const_iterator const iter {methodData_.find(index)};
        if (iter == methodData_.end()) {
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->warn() << "Proloc get reply value cannot find index [" << index.GetProlocInfo() << "]";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            return nullptr;
        }
        typename ReplyData::const_iterator dataIter = iter->second.find(data);
        if (dataIter != iter->second.end()) {
            return dataIter->second;
        }
        /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
        /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
        logInstance_->warn() << "Proloc get unknow reply value [" << index.GetProlocInfo() << "]";
        /* AXIVION enable style AutosarC++19_03-A5.0.1 */
        /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        return nullptr;
    }

private:
    ClientMap methodData_;
    std::mutex clientReplyMutex_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
};



template <typename T>
class ProlocEventMemory : public ProlocMemoryManager {
public:
    using ClientStoreData = std::map<const std::uint8_t*, std::shared_ptr<T>>;
    using ClientDataQueue = std::queue<std::uint8_t*>;
    struct ClientParams {
        ClientStoreData storeDataMap;
        ClientDataQueue dataQueue;
        vrtf::vcc::api::types::EventReceiveHandler handler;
        size_t cacheSize;
    };

    struct SingleClientData {
        std::shared_ptr<std::mutex> clientMutex;
        std::map<ClientUid, ClientParams> singleClientMap;
    };

    ProlocEventMemory()
        : ProlocMemoryManager(), clientMapMutex_ {}, clientMap_ {},
          logInstance_ {ara::godel::common::log::Log::GetLog("CM")} {}
    ~ProlocEventMemory() override {}

    ProlocEventMemory(const ProlocEventMemory& other) = default;
    ProlocEventMemory& operator=(ProlocEventMemory const &prolocEventMemory) = default;

    void StoreData(const T& data, const ProlocEntityIndex& index, const bool isField)
    {
        std::unique_lock<std::mutex> lock {clientMapMutex_};
        // if is filed, first update value
        if (isField) {
            std::shared_ptr<T> dataPtr = std::make_shared<T>(data);
            auto const iter = notifyData_.find(index);
            if (iter == notifyData_.end()) {
                notifyData_.emplace(index, dataPtr);
            } else {
                iter->second = dataPtr;
            }
        }
        typename std::map<ProlocEntityIndex, SingleClientData>::iterator iter {clientMap_.find(index)};
        if (iter == clientMap_.end()) {
            logInstance_->debug() << "[ProlocEventMemory][Can't find event with client][" <<
                index.GetProlocInfo() << "]";
            lock.unlock();
            return;
        }
        std::vector<vrtf::vcc::api::types::EventReceiveHandler> handlerVec;
        {
            std::lock_guard<std::mutex> guard {*iter->second.clientMutex};
            lock.unlock();
            if (iter->second.singleClientMap.empty()) {
                logInstance_->debug() << "[ProlocEventMemory][SingleClientMap is empty, ignore][" <<
                    index.GetProlocInfo() << "]";
                return;
            }
            std::shared_ptr<T> dataPtr {std::make_shared<T>(data)};
            std::uint8_t* rawPtr {reinterpret_cast<std::uint8_t*>(dataPtr.get())};
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->debug() << "Send proloc event[" << index.GetProlocInfo() << "]";
            for (auto clientParamsIter = iter->second.singleClientMap.begin();
                clientParamsIter != iter->second.singleClientMap.end(); ++clientParamsIter)
            {
                AddProlocData(dataPtr, rawPtr, clientParamsIter, index);
                if (clientParamsIter->second.handler != nullptr) {
                    vrtf::vcc::api::types::EventReceiveHandler tmpHandler {clientParamsIter->second.handler};
                    handlerVec.push_back(tmpHandler);
                }
            }
        }
        for (size_t i {0}; i < handlerVec.size(); ++i) {
            handlerVec[i]();
        }
    }

    void EnableEventClient(const ProlocEntityIndex& index, ClientUid const clientUid, const size_t cacheSize,
                           const bool isField, vrtf::vcc::api::types::EventReceiveHandler handler) override
    {
        /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
        /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
        logInstance_->debug() << "Proloc enable client [" << index.GetProlocInfo() << ", clientUid=" <<
            clientUid << "]";
        /* AXIVION enable style AutosarC++19_03-A5.0.1 */
        /* AXIVION enable style AutosarC++19_03-A5.1.1 */
        std::lock_guard<std::mutex> guard {clientMapMutex_};
        typename std::map<ProlocEntityIndex, SingleClientData>::iterator iter {clientMap_.find(index)};
        ClientStoreData emptyMap;
        ClientDataQueue emptyQueue;

        ClientParams clientParams {emptyMap, emptyQueue, handler, cacheSize};
        bool triggerHandler = false;
        if (isField) {
            typename std::map<ProlocEntityIndex, std::shared_ptr<T>>::const_iterator notifyIter = notifyData_.find(index);
            if (notifyIter != notifyData_.cend()) {
                std::shared_ptr<T> dataTmp = notifyIter->second;
                std::uint8_t* rawPtr {reinterpret_cast<std::uint8_t*>(dataTmp.get())};
                clientParams.storeDataMap.emplace(rawPtr, dataTmp);
                clientParams.dataQueue.push(rawPtr);
                triggerHandler = true;
            } else {
                logInstance_->warn() << "Proloc field not have a init value client [" << index.GetProlocInfo() <<
                    ", clientUid=" << clientUid << "]";
            }
        }
        if (iter == clientMap_.end()) {
            std::map<ClientUid, ClientParams> initClientMap;
            static_cast<void>(initClientMap.emplace(clientUid, clientParams));
            std::shared_ptr<std::mutex> mutexPtr {std::make_shared<std::mutex>()};
            SingleClientData singClientData {mutexPtr, initClientMap};
            static_cast<void>(clientMap_.emplace(index, singClientData));
        } else {
            {
                std::lock_guard<std::mutex> guardTmp {*iter->second.clientMutex};
                static_cast<void>(iter->second.singleClientMap.emplace(clientUid, clientParams));
            }
        }
        if (triggerHandler) {
            handler();
        }
    }

    void SetReceiveHandler(vrtf::vcc::api::types::EventReceiveHandler handler, ClientUid const clientUid,
                           ProlocEntityIndex index) override
    {
        std::unique_lock<std::mutex> lock {clientMapMutex_};
        typename std::map<ProlocEntityIndex, SingleClientData>::iterator iter {clientMap_.find(index)};
        if (iter == clientMap_.end()) {
            lock.unlock();
            return;
        }
        {
            std::lock_guard<std::mutex> guard {*iter->second.clientMutex};
            lock.unlock();
            typename std::map<ClientUid, ClientParams>::iterator clientIter {
                iter->second.singleClientMap.find(clientUid)};
            ClientParams params;
            if (clientIter == iter->second.singleClientMap.end()) {
                params.handler = handler;
                static_cast<void>(iter->second.singleClientMap.emplace(clientUid, params));
            } else {
                clientIter->second.handler = handler;
            }
        }
    }

    size_t GetMessageNumber(const vrtf::driver::proloc::ProlocEntityIndex& index,
                                    ClientUid const clientUid) noexcept override
    {
        std::unique_lock<std::mutex> lock {clientMapMutex_};
        typename std::map<ProlocEntityIndex, SingleClientData>::iterator iter {clientMap_.find(index)};
        if (iter == clientMap_.end()) {
            lock.unlock();
            return 0;
        }
        {
            std::lock_guard<std::mutex> guard{*iter->second.clientMutex};
            lock.unlock();
            typename std::map<ClientUid, ClientParams>::iterator clientIter{
                iter->second.singleClientMap.find(clientUid)};
            if (clientIter == iter->second.singleClientMap.end()) {
                return 0;
            }
            return clientIter->second.dataQueue.size();
        }
    }

    std::vector<std::uint8_t*> ReadProlocEvent(const vrtf::driver::proloc::ProlocEntityIndex& index,
        ClientUid const clientUid, const std::int32_t size) override
    {
        std::unique_lock<std::mutex> lock {clientMapMutex_};
        std::vector<std::uint8_t*> dataVec;
        typename std::map<ProlocEntityIndex, SingleClientData>::iterator iter {clientMap_.find(index)};
        if (iter == clientMap_.end()) {
            lock.unlock();
            return dataVec;
        }
        {
            std::lock_guard<std::mutex> guard {*iter->second.clientMutex};
            lock.unlock();
            typename std::map<ClientUid, ClientParams>::iterator clientIter {
                iter->second.singleClientMap.find(clientUid)};
            if (clientIter == iter->second.singleClientMap.end()) {
                return dataVec;
            }
            ClientDataQueue& dataQueue {clientIter->second.dataQueue};
            for (int32_t i {0}; i < size; ++i) {
                if (dataQueue.size() != 0) {
                    std::uint8_t* data {dataQueue.front()};
                    dataVec.push_back(data);
                    dataQueue.pop();
                } else {
                    break;
                }
            }
        }
        return dataVec;
    }

    void ReturnLoan(const vrtf::driver::proloc::ProlocEntityIndex& index, ClientUid const clientUid,
                    const std::uint8_t* data) override
    {
        std::unique_lock<std::mutex> lock {clientMapMutex_};
        typename std::map<ProlocEntityIndex, SingleClientData>::iterator indexIter {clientMap_.find(index)};
        if (indexIter == clientMap_.end()) {
            lock.unlock();
            return;
        }

        {
            std::lock_guard<std::mutex> guard {*indexIter->second.clientMutex};
            lock.unlock();
            typename std::map<ClientUid, ClientParams>::iterator clientIter {
                indexIter->second.singleClientMap.find(clientUid)};
            if (clientIter == indexIter->second.singleClientMap.end()) {
                return;
            }
            ClientStoreData& storeData {clientIter->second.storeDataMap};
            typename ClientStoreData::const_iterator dataIter {storeData.find(data)};
            if (dataIter != storeData.cend()) {
                static_cast<void>(storeData.erase(dataIter));
                logInstance_->debug() << "Event release memory successful[" << index.GetProlocInfo() <<
                    ", clientUid=" << clientUid << "]";
            } else {
                /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
                /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
                logInstance_->warn("ProlocRequestMemory_ReturnLoan", {vrtf::vcc::api::types::DEFAULT_LOG_LIMIT,
                    ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                    "Event [" << index.GetProlocInfo() << ", clientUid=" << clientUid << "] release memory failed";
                /* AXIVION enable style AutosarC++19_03-A5.0.1 */
                /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            }
        }
    }

    std::shared_ptr<T> GetSharedPtr(const ProlocEntityIndex& index, const std::uint8_t* data,
                                    ClientUid const clientUid)
    {
        std::unique_lock<std::mutex> lock {clientMapMutex_};
        typename std::map<ProlocEntityIndex, SingleClientData>::const_iterator iter {clientMap_.find(index)};
        if (iter == clientMap_.end()) {
            lock.unlock();
            return nullptr;
        }

        {
            std::lock_guard<std::mutex> guard {*iter->second.clientMutex};
            lock.unlock();
            typename std::map<ClientUid, ClientParams>::const_iterator clientIter {
                iter->second.singleClientMap.find(clientUid)};
            if (clientIter == iter->second.singleClientMap.end()) {
                return nullptr;
            }

            const ClientStoreData& storeData {clientIter->second.storeDataMap};
            typename ClientStoreData::const_iterator dataIter {storeData.find(data)};
            if (dataIter != storeData.end()) {
                return dataIter->second;
            }
        }
        return nullptr;
    }

    void UnSubscribeClient(const ProlocEntityIndex& index, ClientUid const clientUid) override
    {
        std::shared_ptr<std::mutex> holdClientMutex;
        {
            std::lock_guard<std::mutex> guard {clientMapMutex_};
            typename std::map<ProlocEntityIndex, SingleClientData>::iterator iter {clientMap_.find(index)};
            if (iter == clientMap_.end()) {
                return;
            }
            std::lock_guard<std::mutex> clientGuard {*iter->second.clientMutex};
            holdClientMutex = iter->second.clientMutex;
            typename std::map<ClientUid, ClientParams>::const_iterator clientIter
                {iter->second.singleClientMap.find(clientUid)};
            if (clientIter == iter->second.singleClientMap.end()) {
                return;
            }
            /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
            /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
            logInstance_->verbose() << "Proloc event client [" << index.GetProlocInfo() << ", clientUid=" << clientUid <<
                "] erase successful";
            /* AXIVION enable style AutosarC++19_03-A5.0.1 */
            /* AXIVION enable style AutosarC++19_03-A5.1.1 */
            static_cast<void>(iter->second.singleClientMap.erase(clientIter));
            if (iter->second.singleClientMap.size() == 0) {
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance_->verbose() << "Proloc event client [" << index.GetProlocInfo() << "] all erase successful";
                static_cast<void>(clientMap_.erase(iter));
            }
        }
    }

    void StoreRawMemoryData(vrtf::vcc::api::types::RawBuffer& rawBuffer, const ProlocEntityIndex& index)
    {
        using namespace vrtf::vcc::api::types;
        std::unique_lock<std::mutex> lock {clientMapMutex_};
        typename std::map<ProlocEntityIndex, SingleClientData>::iterator iter {clientMap_.find(index)};
        if (iter == clientMap_.end()) {
            lock.unlock();
            return;
        }
        std::vector<vrtf::vcc::api::types::EventReceiveHandler> handlerContainer;
        {
            std::lock_guard<std::mutex> guard {*iter->second.clientMutex};
            lock.unlock();
            if (iter->second.singleClientMap.empty()) {
                return;
            }
            size_t const size = rawBuffer.GetRawBufferSize();
            std::shared_ptr<std::pair<std::vector<std::uint8_t>, size_t>> dataPtr =
                    std::make_shared<std::pair<std::vector<std::uint8_t>, size_t>>(
                            std::make_pair(std::vector<std::uint8_t>(size), size));
            std::uint8_t* rawPtr = dataPtr->first.data();
            auto memcpySuccess = memcpy_s(rawPtr, dataPtr->first.size(), rawBuffer.GetRawBufferPtr(), size);
            if (memcpySuccess != 0) {
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance_->error("ProlocRequestMemory_StoreRawMemoryData",
                {DEFAULT_LOG_LIMIT, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                    "[ProlocEventMemory][Proloc rawmemory memcpy failed][prolocInfo=" << index.GetProlocInfo() << "]";
                return;
            }
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->debug() << "Send proloc event[" << index.GetProlocInfo() << "]";
            for (auto clientParamsIter = iter->second.singleClientMap.begin();
                 clientParamsIter != iter->second.singleClientMap.end(); ++clientParamsIter)
            {
                AddProlocData(dataPtr, rawPtr, clientParamsIter, index);
                if (clientParamsIter->second.handler != nullptr) {
                    vrtf::vcc::api::types::EventReceiveHandler tmpHandler {clientParamsIter->second.handler};
                    handlerContainer.push_back(tmpHandler);
                }
            }
        }
        for (size_t i {0}; i < handlerContainer.size(); ++i) {
            handlerContainer[i]();
        }
    }

private:
    void AddProlocData(std::shared_ptr<T> dataPtr, std::uint8_t* rawPtr,
        const typename std::map<ClientUid, ClientParams>::iterator& clientParamsIter, const ProlocEntityIndex& index)
    {
        static_cast<void>(clientParamsIter->second.storeDataMap.emplace(rawPtr, dataPtr));
        if (clientParamsIter->second.dataQueue.size() == clientParamsIter->second.cacheSize) {
            std::uint8_t* returnRawPtr {clientParamsIter->second.dataQueue.front()};
            logInstance_->warn("ProlocRequestMemory_AddProlocData", {vrtf::vcc::api::types::DEFAULT_LOG_LIMIT,
            ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                "[ProlocEventMemory][Proloc event msg is full, oldest data wii be discard][prolocInfo=" <<
                index.GetProlocInfo() << ", clientUid=" << clientParamsIter->first << "]";
            clientParamsIter->second.dataQueue.pop();
            typename ClientStoreData::const_iterator rawPtrIter {
                    clientParamsIter->second.storeDataMap.find(returnRawPtr)};
            if (rawPtrIter != clientParamsIter->second.storeDataMap.end()) {
                static_cast<void>(clientParamsIter->second.storeDataMap.erase(rawPtrIter));
            }
            clientParamsIter->second.dataQueue.push(rawPtr);
        } else {
            clientParamsIter->second.dataQueue.push(rawPtr);
        }
    }

    std::mutex clientMapMutex_;
    std::map<ProlocEntityIndex, SingleClientData> clientMap_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    // just use for field notify when new client online
    std::map<ProlocEntityIndex, std::shared_ptr<T>> notifyData_;
};

class ProlocMemoryFactory final {
public:
    ProlocMemoryFactory() = default;
    ~ProlocMemoryFactory() = default;
    ProlocMemoryFactory(const ProlocMemoryFactory& other) = default;
    ProlocMemoryFactory& operator=(ProlocMemoryFactory const &prolocMemoryFactory) = default;
    template <typename T>
    static std::shared_ptr<ProlocEventMemory<T>>& GetEventInstance() noexcept
    {
        static std::shared_ptr<ProlocEventMemory<T>> singleInstance {std::make_shared<ProlocEventMemory<T>>()};
        return singleInstance;
    }

    template <class... Args>
    static std::shared_ptr<ProlocRequestMemory<Args...>>& GetMethodRequestInstance() noexcept
    {
        static std::shared_ptr<ProlocRequestMemory<Args...>> singleInstance {
            std::make_shared<ProlocRequestMemory<Args...>>()};
        return singleInstance;
    }

    template <class ReplyType>
    static std::shared_ptr<ProlocReplyMemory<ReplyType>>& GetMethodReplyInstance() noexcept
    {
        static std::shared_ptr<ProlocReplyMemory<ReplyType>> singleInstance {
            std::make_shared<ProlocReplyMemory<ReplyType>>()};
        return singleInstance;
    }

};
}
}
}
#endif // INC_ARA_VCC_DRIVER_DDS_DRIVER_HPP_
