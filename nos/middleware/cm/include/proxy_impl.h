#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>

#include "cm/include/participant_factory.h"

namespace hozon {
namespace netaos {
namespace cm {

class ProxyImpl {
   public:
    ProxyImpl(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type);
    ProxyImpl(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type, uint8_t qos_mode);

    ~ProxyImpl();
    int32_t Init(const uint32_t domain, const std::string& topic);
    void Deinit();

    int32_t Take(std::shared_ptr<void> data, uint64_t timeout_us);
    int32_t Take(std::shared_ptr<void> data);

    template <class T>
    int32_t Take(const std::function<void(const T&)>& do_take) {
        using TSeq = eprosima::fastdds::dds::LoanableSequence<T>;
        TSeq data;
        SampleInfoSeq infos;

        if (ReturnCode_t::RETCODE_OK == _reader->take(data, infos, 1)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                if (infos[i].valid_data) {
                    // Print the data.
                    const T& sample = data[i];
                    do_take(sample);
                    _reader->return_loan(data, infos);
                }
            }
        }

        return 0;
    }

    using DataAvailableCallback = std::function<void(void)>;
    void Listen(DataAvailableCallback callback);

    bool IsMatched();

   private:
    void InitListenerProcess(void);
    void DeinitListenerProcess(void);

    void LogReaderQosInfo(eprosima::fastdds::dds::DataReader* reader);
    void LogStatisticInfo(void);

    eprosima::fastdds::dds::DataReader* _reader;
    eprosima::fastdds::dds::Topic* _topic_desc;
    eprosima::fastdds::dds::TypeSupport _type;

    class SubListener : public eprosima::fastdds::dds::DataReaderListener {
       public:
        SubListener() = default;
        ~SubListener() override = default;

        void on_data_available(eprosima::fastdds::dds::DataReader* reader) override;

        void on_subscription_matched(eprosima::fastdds::dds::DataReader* reader, const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;

        int matched = 0;

        uint32_t _domain;
        std::string _topic;

        DataAvailableCallback callback = nullptr;
        std::mutex _mtx;
        std::condition_variable _cv;
        bool new_data_arrived = false;
        uint64_t callback_total_count = 0;
        uint64_t history_overflow_count = 0;
    } _listener;

    uint32_t _domain;
    std::string _topic;

    std::thread _listener_thread;
    bool _need_stop = false;
    QosMode _qos_mode = NO_MODE;

    struct ProxyStatisticInfo {
        uint64_t cm_callback;
        uint64_t user_callback;
        uint64_t from_protocol;
        uint64_t from_protocol_bytes;
        uint64_t to_user;
        uint64_t sample_lost;
        uint64_t history_overflow;
    };

    ProxyStatisticInfo _statistic_info = {0};
};

}  // namespace cm
}  // namespace netaos
}  // namespace hozon
