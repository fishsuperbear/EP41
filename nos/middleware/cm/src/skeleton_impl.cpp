#include "cm/include/skeleton_impl.h"
#include "cm/include/cm_logger.h"

namespace hozon {
namespace netaos {
namespace cm {

using namespace eprosima::fastdds::dds;

SkeletonImpl::SkeletonImpl(std::shared_ptr<TopicDataType> topic_type)
    : _writer(nullptr)
    , _topic_desc(nullptr)
    , _type(topic_type) 
    , _domain(0)
    , _topic("skeleton") {
}

SkeletonImpl::SkeletonImpl(std::shared_ptr<TopicDataType> topic_type, uint8_t qos_mode)
    : _writer(nullptr)
    , _topic_desc(nullptr)
    , _type(topic_type) 
    , _domain(0)
    , _topic("skeleton")
    , _qos_mode((QosMode)qos_mode) {
}

SkeletonImpl::~SkeletonImpl() {
}

void SkeletonImpl::LogStatisticInfo(void) {
    _statistic_info.to_protocol_bytes = _statistic_info.to_protocol * _type->m_typeSize;

    CM_LOG_INFO_WITH_HEAD << "from_user : " << _statistic_info.from_user;
    CM_LOG_INFO_WITH_HEAD << "to_protocol : " << _statistic_info.to_protocol;
    CM_LOG_INFO_WITH_HEAD << "to_protocol_bytes : " << _statistic_info.to_protocol_bytes;

    memset(&_statistic_info, 0x00, sizeof(_statistic_info));
}

void SkeletonImpl::Deinit() {
    CM_LOG_INFO_WITH_HEAD << "Skeleton deinit.";

    LogStatisticInfo();
    DdsPubSubInstance::GetInstance().DeleteWriter(_domain, _writer);
    DdsPubSubInstance::GetInstance().DeleteTopicDesc(_domain, _topic_desc);
    DdsPubSubInstance::GetInstance().DeletePublisher(_domain);
    DdsPubSubInstance::GetInstance().DeleteParticpant(_domain);
}

int32_t SkeletonImpl::Init(const uint32_t domain, const std::string& topic) {
    int ret = -1;

    _domain = domain;
    _topic = topic;
    _listener._domain = _domain;
    _listener._topic = topic;

    if ((topic == "/soc/pointcloud") || (topic == "/soc/rawpointcloud")) {
        _domain = 1;
    }

    CM_LOG_INFO_WITH_HEAD << "Skeleton init start.";

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

    Publisher* publisher = DdsPubSubInstance::GetInstance().GetPublisher(_domain);
    if (publisher == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Create publisher failed.";
        return -1;
    }

    _topic_desc = DdsPubSubInstance::GetInstance().GetTopicDescription(_domain, topic, _type.get_type_name());
    if (_topic_desc == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Create topic failed.";
        return -1;
    }

    //auto data_type_ptr = _type.create_data();

    _writer = DdsPubSubInstance::GetInstance().GetDataWriter(_domain, _topic_desc, &_listener);
    if (_writer == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Create writer failed.";
        return -1;
    }

    CM_LOG_INFO_WITH_HEAD << "Writer : '" << convertGuidToString(_writer->guid()) << "' created.";

    //_type.delete_data(data_type_ptr);

    LogWriterQosInfo(_writer);

    CM_LOG_INFO_WITH_HEAD << "Skeleton init success.";
    return 0;
}

void SkeletonImpl::LogWriterQosInfo(DataWriter* writer) {
    CM_LOG_INFO_WITH_HEAD << "Writer qos info : "
            << " reliability : " << writer->get_qos().reliability().kind
            << " durability : " << writer->get_qos().durability().kind
            << " history.kind : " << writer->get_qos().history().kind
            << " history.depth : " << writer->get_qos().history().depth
            << " history_memory_policy : " << writer->get_qos().endpoint().history_memory_policy
            << " data_sharing : " << writer->get_qos().data_sharing().kind()
            << " usr_qos_mode : " << _qos_mode;
}

int32_t SkeletonImpl::Write(std::shared_ptr<void> data) {
    if (_writer == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Skeleton writer is not create.";
        return -1;
    }

    _statistic_info.from_user++;

    bool ret = _writer->write(data.get());
    if (ret == true) {
        CM_LOG_TRACE_WITH_HEAD << "Write data is successful.";
        _statistic_info.to_protocol++;
        return 0;
    }
    else {
        CM_LOG_ERROR_WITH_HEAD << "Write data failed.";
        return -1;
    }
}

void SkeletonImpl::PubListener::on_publication_matched(DataWriter* writer,
        const PublicationMatchedStatus& info) {
    if (info.current_count_change == 1) {
        matched = info.current_count;
        CM_LOG_INFO_WITH_HEAD << "Skeleton matched." << " current count : " << matched;

        if (service_find_callback) {
            service_find_callback();
        }
    }
    else if (info.current_count_change == -1) {
        matched = info.current_count;
        CM_LOG_INFO_WITH_HEAD << "Skeleton unmatched." << " current count : " << matched;
    }
    else {
        CM_LOG_INFO_WITH_HEAD << info.current_count_change
                  << "Skeleton count change is not a valid value.";
    }
}

void SkeletonImpl::RegisterServiceListen(OnServiceFindCallback callback) {
    _listener.service_find_callback = callback;
    if (_listener.service_find_callback == nullptr) {
        CM_LOG_ERROR_WITH_HEAD << "Service find callback function is NULL.";
    }

    CM_LOG_DEBUG_WITH_HEAD << "Register service find callback success.";
}

bool SkeletonImpl::IsMatched() {
    if (_listener.matched == 0 ) {
        return false;
    }
    else {
        return true;
    }
}


}
}
}