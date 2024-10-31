#pragma once

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>

#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/rtps/transport/UDPv6TransportDescriptor.h>
#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/PublisherAttributes.h>
#include <fastdds/rtps/resources/ResourceManagement.h>

#include <unordered_map>

namespace hozon {
namespace netaos {
namespace cm {

#define QOS_CONFIG_PATH "NETA_CM_QOS_PATH"
#define DEFAULT_QOS_TOPIC "DefaultTopic"
#define DEFAULT_CONFIG_PATH "CONF_PREFIX_PATH"

const std::vector<std::string> default_network_List = {
    "127.0.0.1",    // x86 loopback
    "192.168.33.42",
    "192.168.33.110"
};

enum QosMode {
    NO_MODE = 0,
    METHOD_MODE = 1
};

struct endpointInfo {
    std::string history_memory_policy;
};

struct historyInfo {
    std::string kind;
    uint32_t depth;
};

struct DataReaderQosInfo {
    std::string topic;
    std::string reliability;
    std::string durability;
    struct endpointInfo endpoint;
    struct historyInfo history;
    std::string data_sharing;
};

struct DataWriterQosInfo {
    std::string topic;
    std::string reliability;
    std::string durability;
    struct endpointInfo endpoint;
    struct historyInfo history;
    std::string data_sharing;
};

struct DiscoveryQosInfo {
    bool typelookup_client;
    bool typelookup_server;
    uint32_t leaseDuration;
    uint32_t leaseDuration_announce_period;
    uint32_t initial_announce_count;
    uint32_t initial_announce_period;
};

struct UdpQosInfo {
    bool enable;
    std::string network;
    uint32_t send_socket_buffer_size;
    uint32_t listen_socket_buffer_size;
};

struct ShmQosInfo {
    bool enable;
    uint32_t segment_size;
    uint32_t max_message_size;
};

struct TransportQosInfo {
    bool use_builtin_transports;
    UdpQosInfo udp_qos_info;
    ShmQosInfo shm_qos_info;
};

struct ParticipantQosInfo {
    std::string name;
    uint32_t domain_id;
    TransportQosInfo transport;
    DiscoveryQosInfo discover;
    std::unordered_map<std::string, DataReaderQosInfo> dataReaderInfo;
    std::unordered_map<std::string, DataWriterQosInfo> dataWriterInfo;
};

class CmQosConfig {
public:
    int32_t GetQosFromEnv(const std::string env, const uint32_t domain);
    int32_t GetQosFromEnv(const std::string env, const uint32_t domain, const std::string file_name);
    eprosima::fastdds::dds::DomainParticipantQos GetParticipantQos(const uint32_t domain);
    eprosima::fastdds::dds::DataWriterQos GetWriterQos(const uint32_t domain, const std::string& topic);
    eprosima::fastdds::dds::DataReaderQos GetReaderQos(const uint32_t domain, const std::string& topic);
    eprosima::fastdds::dds::DataReaderQos GetReaderQos(const uint32_t domain, const std::string& topic, QosMode mode);

private:
    std::mutex _mutex;
    std::unordered_map<uint32_t, ParticipantQosInfo> _qos_config_map;

    int32_t GetNetworkList(std::vector<std::string>& network_List);
    int32_t ReadJsonFile(const std::string &filePath, std::unordered_map<uint32_t, ParticipantQosInfo> &participantparQosInfoMap);

    void MappingParticipantQos(const ParticipantQosInfo& participantQosInfo, eprosima::fastdds::dds::DomainParticipantQos& participantQos);
    void MappingDataWriterQos(const DataWriterQosInfo& dataWriterQosInfo, eprosima::fastdds::dds::DataWriterQos& dataWriterQos);
    void MappingDataReaderQos(const DataReaderQosInfo& dataReaderQosInfo, eprosima::fastdds::dds::DataReaderQos& dataReaderQos);
};

}
}
}