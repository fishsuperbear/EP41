#include "cm/include/proxy_impl.h"
#include "cm/include/cm_logger.h"

namespace hozon {
namespace netaos {
namespace cm {

using namespace eprosima::fastdds::dds;

ProxyImpl::ProxyImpl(std::shared_ptr<TopicDataType> topic_type)
    : _reader(nullptr)
    , _topic_desc(nullptr)
    , _type(topic_type)
    , _domain(0)
    , _topic("proxy") {
}

ProxyImpl::ProxyImpl(std::shared_ptr<TopicDataType> topic_type, uint8_t qos_mode)
    : _reader(nullptr)
    , _topic_desc(nullptr)
    , _type(topic_type)
    , _domain(0)
    , _topic("proxy")
    , _qos_mode((QosMode)qos_mode) {
}

ProxyImpl::~ProxyImpl() {
}

void ProxyImpl::LogStatisticInfo(void) {
    _statistic_info.cm_callback = _listener.callback_total_count;
    _statistic_info.from_protocol = _statistic_info.cm_callback;
    _statistic_info.from_protocol_bytes = _statistic_info.from_protocol * _type->m_typeSize;
    _statistic_info.history_overflow = _listener.history_overflow_count;

    SampleLostStatus status;
    _reader->get_sample_lost_status(status);
    _statistic_info.sample_lost = status.total_count;

    CM_LOG_INFO_WITH_HEAD << "cm_callback : " << _statistic_info.cm_callback;
    CM_LOG_INFO_WITH_HEAD << "user_callback : " << _statistic_info.user_callback;
    CM_LOG_INFO_WITH_HEAD << "from_protocol : " << _statistic_info.from_protocol;
    CM_LOG_INFO_WITH_HEAD << "from_protocol_bytes : " << _statistic_info.from_protocol_bytes;
    CM_LOG_INFO_WITH_HEAD << "to_user : " << _statistic_info.to_user;
    CM_LOG_INFO_WITH_HEAD << "sample_lost : " << _statistic_info.sample_lost;
    CM_LOG_INFO_WITH_HEAD << "history_overflow : " << _statistic_info.history_overflow;

    memset(&_statistic_info, 0x00, sizeof(_statistic_info));
}

void ProxyImpl::Deinit() {
    CM_LOG_INFO_WITH_HEAD << "Proxy deinit. ";

    DeinitListenerProcess();
    LogStatisticInfo();
    DdsPubSubInstance::GetInstance().DeleteReader(_domain, _reader);
    DdsPubSubInstance::GetInstance().DeleteTopicDesc(_domain, _topic_desc);
    DdsPubSubInstance::GetInstance().DeleteSublisher(_domain);
    DdsPubSubInstance::GetInstance().DeleteParticpant(_domain);
}

int32_t ProxyImpl::Init(const uint32_t domain, const std::string& topic) {
    int ret = -1;

    _domain = domain;
    _topic = topic;
    _listener._domain = _domain;
    _listener._topic = topic;
    _need_stop = false;

    if ((topic == "/soc/pointcloud") || (topic == "/soc/rawpointcloud")) {
        _domain = 1;
    }

    CM_LOG_INFO_WITH_HEAD << "Proxy init start. ";

    DomainParticipant* participant = DdsPubSubInstance::GetInstance().Getparticipant(_domain);
    if (participant == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Create participant failed.";
        return -1;
    }

    CM_LOG_INFO_WITH_HEAD << "Participant : '" << convertGuidToString(participant->guid()) << "' created.";

    ret = DdsPubSubInstance::GetInstance().RegisterTopicType(_domain, _type);
    if (ret != ReturnCode_t::RETCODE_OK) {
        CM_LOG_ERROR_WITH_HEAD << "Register type failed.";
        return -1;
    }

    Subscriber* subscriber = DdsPubSubInstance::GetInstance().GetSubscriber(_domain);
    if (subscriber == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Create sublisher failed.";
        return -1;
    }

    _topic_desc = DdsPubSubInstance::GetInstance().GetTopicDescription(_domain, topic, _type.get_type_name());
    if (_topic_desc == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Create topic failed.";
        return -1;
    }

    _reader = DdsPubSubInstance::GetInstance().GetDataReader(_domain, _topic_desc, &_listener, _qos_mode);
    if (_reader == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Create reader failed.";
        return -1;
    }

    CM_LOG_INFO_WITH_HEAD << "Reader : '" << convertGuidToString(_reader->guid()) << "' created.";

    InitListenerProcess();
    LogReaderQosInfo(_reader);

    CM_LOG_INFO_WITH_HEAD << "Proxy init success.";
    return 0;
}

void ProxyImpl::LogReaderQosInfo(DataReader* reader) {
    CM_LOG_INFO_WITH_HEAD << "Reader qos info : "
            << " reliability : " << reader->get_qos().reliability().kind
            << " durability : " << reader->get_qos().durability().kind
            << " history.kind : " << reader->get_qos().history().kind
            << " history.depth : " << reader->get_qos().history().depth
            << " history_memory_policy : " << reader->get_qos().endpoint().history_memory_policy
            << " data_sharing : " << reader->get_qos().data_sharing().kind()
            << " usr_qos_mode : " << _qos_mode;
}

void ProxyImpl::SubListener::on_subscription_matched(DataReader* reader,
        const SubscriptionMatchedStatus& info) {
    if (info.current_count_change == 1) {
        matched = info.current_count;
        CM_LOG_INFO_WITH_HEAD << "Proxy matched." << " current count : " << matched;
    }
    else if (info.current_count_change == -1) {
        matched = info.current_count;
        CM_LOG_INFO_WITH_HEAD << "Proxy unmatched." << " current count : " << matched;
    }
    else {
        CM_LOG_INFO_WITH_HEAD<< info.current_count_change
                  << "Proxy count change is not a valid value.";
    }
}

int32_t ProxyImpl::Take(std::shared_ptr<void> data) {
    if (_reader == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Take reader is not create.";
        return -1;
    }

    SampleInfo info;
    ReturnCode_t ret = _reader->take_next_sample(data.get(), &info);
    if (ret == ReturnCode_t::RETCODE_OK) {
        if (info.valid_data) {
            CM_LOG_TRACE_WITH_HEAD << "Take data is successful";
            _statistic_info.to_user++;
            return 0;
        }
        else {
            CM_LOG_ERROR_WITH_HEAD << "Take data is invalid. errcode : " << ret();
            return -1;
        }
    }
    else {
        CM_LOG_ERROR_WITH_HEAD << "Take no data. errcode : " << ret();
        return -1;
    }

    return 0;
}

int32_t ProxyImpl::Take(std::shared_ptr<void> data, uint64_t timeout_us) {
    if (_reader == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Take(timeout) reader is not create.";
        return -1;
    }

    int32_t seconds = timeout_us / 1000 / 1000;
    uint32_t nanosec = (timeout_us - seconds * 1000 * 1000) * 1000;

    Duration_t wait_time(seconds, nanosec);
    if (_reader->wait_for_unread_message(wait_time)) {
        return Take(data);
    }
    else {
        CM_LOG_TRACE_WITH_HEAD << "Proxy receive data timeout(us): " << timeout_us;
        return -1;
    }
    return 0;
}

void ProxyImpl::Listen(DataAvailableCallback callback) {
    _listener.callback = callback;
    if (_listener.callback == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Listen callback function is NULL.";
    }

    CM_LOG_DEBUG_WITH_HEAD << "Register listener callback success.";
}

void ProxyImpl::SubListener::on_data_available(
        DataReader* reader) {
    callback_total_count++;
    if (callback) {
        std::unique_lock<std::mutex> lck(_mtx);
        new_data_arrived = true;
        _cv.notify_all();
    }
}

bool ProxyImpl::IsMatched() {
    if (_listener.matched == 0 ) {
        return false;
    }
    else {
        return true;
    }
}

void ProxyImpl::DeinitListenerProcess(void) {
    {
        std::unique_lock<std::mutex> lck(_listener._mtx);
        _need_stop = true;
        _listener.new_data_arrived = true;
        _listener._cv.notify_all();
    }
    if (_listener_thread.joinable()) {
        _listener_thread.join();
    }
}

void ProxyImpl::InitListenerProcess(void) {
    _listener_thread = std::thread([this]() {
        while (!_need_stop) {
            {
                std::unique_lock<std::mutex> lck(_listener._mtx);
                _listener._cv.wait(lck, [&](){return _listener.new_data_arrived;});
                _listener.new_data_arrived = false;

                if (_need_stop) {
                    return;
                }
            }

            if (_listener.callback == nullptr) {
                CM_LOG_TRACE_WITH_HEAD << "Proxy listern callback is null.";
                continue;
            }

            while(_reader->get_unread_count() > 0) {
                SubscriptionMatchedStatus status;
                _reader->get_subscription_matched_status(status);
                if (status.current_count == 0) {
                    CM_LOG_INFO_WITH_HEAD << "Proxy no matched status.";
                    break;
                }

                if (_need_stop) {
                    break;
                }

                CM_LOG_TRACE_WITH_HEAD << "Start proxy callback function."
                        << "total arrived : " << _reader->get_unread_count();
                _statistic_info.user_callback++;
                _listener.callback();
            }
        }
    });
    pthread_setname_np(_listener_thread.native_handle(), (std::string("cm_") + _topic.substr(0,11)).c_str());
}

}
}
}