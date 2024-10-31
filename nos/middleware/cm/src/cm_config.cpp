#include <fstream>
#include <unistd.h>
#include <sys/param.h>
#include <iostream>
#include <string>
#include <json/json.h>

#include "cm/include/cm_config.h"
#include "cm/include/cm_logger.h"

namespace hozon {
namespace netaos {
namespace cm {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;
using namespace eprosima::fastrtps::rtps;

#define CM_RELIABILITY_RELIABLE     0
#define CM_RELIABILITY_BEST_EFFORT  1

#define CM_DURABILITY_VOLATILE          0
#define CM_DURABILITY_TRANSIENT_LOCAL   1
#define CM_DURABILITY_TRANSIENT         2

#define CM_HISTORY_KEEP_LAST        0
#define CM_HISTORY_KEEP_ALL         1

#define CM_HISTORY_KIND_RELIABLE    0
#define CM_HISTORY_KIND_BEST_EFFORT 1

#define CM_DATA_SHARE_AUTO          0
#define CM_DATA_SHARE_ON            1
#define CM_DATA_SHARE_OFF           2

#define CM_PREALLOCATED                  0
#define CM_PREALLOCATED_WITH_REALLOC     1
#define CM_DYNAMIC_RESERVE               2
#define CM_DYNAMIC_REUSABLE              3

std::unordered_map<std::string, uint8_t> cm_qos_reliability = {
    {"RELIABLE",     CM_RELIABILITY_RELIABLE},
    {"BEST_EFFORT",  CM_RELIABILITY_BEST_EFFORT},
};

std::unordered_map<std::string, uint8_t> cm_qos_durability = {
    {"VOLATILE_DURABILITY_QOS",            CM_DURABILITY_VOLATILE},
    {"TRANSIENT_LOCAL_DURABILITY_QOS",     CM_DURABILITY_TRANSIENT_LOCAL},
    {"TRANSIENT_DURABILITY_QOS",           CM_DURABILITY_TRANSIENT},
};

std::unordered_map<std::string, uint8_t> cm_qos_history_kind = {
    {"KEEP_LAST",    CM_HISTORY_KEEP_LAST},
    {"KEEP_ALL",     CM_HISTORY_KEEP_ALL},
};

std::unordered_map<std::string, uint8_t> cm_qos_data_sharing = {
    {"AUTO",         CM_DATA_SHARE_AUTO},
    {"OFF",          CM_DATA_SHARE_OFF},
    {"ON",           CM_DATA_SHARE_ON},
};

std::unordered_map<std::string, uint8_t> cm_qos_memory_policy = {
    {"PREALLOCATED_MEMORY_MODE",                CM_PREALLOCATED},
    {"PREALLOCATED_WITH_REALLOC_MEMORY_MODE",   CM_PREALLOCATED_WITH_REALLOC},
    {"DYNAMIC_RESERVE_MEMORY_MODE",             CM_DYNAMIC_RESERVE},
    {"DYNAMIC_REUSABLE_MEMORY_MODE",            CM_DYNAMIC_REUSABLE},
};

std::string ReadFile(const std::string &filePath)
{
    char absolutePath[PATH_MAX] {};
    char *ret = realpath(filePath.data(), absolutePath);
    if (ret == nullptr) {
        return "";
    }
    std::ifstream in(absolutePath);
    if (!in.good()) {
        return "";
    }
    std::string contents { std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
    return contents;
}

int32_t OpenJsonFile(const std::string &filePath, Json::Value &root)
{
    std::string fileContent = ReadFile(filePath);
    if (fileContent.empty()) {
        return -1;
    }
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    const int rawJsonLength = static_cast<int>(fileContent.length());
    std::string err;
    if (!reader->parse(fileContent.c_str(), fileContent.c_str() + rawJsonLength, &root, &err)) {
        return -1;
    }

    return 0;
}

void ParseJson(bool &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return;
    }

    if (!jsonValue.isBool()) {
        CM_LOG_ERROR << ("Failed to convert the value of the " + key + " key to the [bool] type.");
        return;
    }

    elem = jsonValue.asBool();
}

int32_t ParseJson(std::vector<std::string>& elem, const Json::Value& currentNode, const std::string& key) {
    Json::Value defaultValue;
    const auto& jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    if (!jsonValue.isArray()) {
        CM_LOG_ERROR << ("Failed to convert the value of the " + key + " key to the [vector<std::string>] type.");
        return -1;
    }

    for (auto item : jsonValue) {
        elem.push_back(item.asString());
    }

    return 0;
}

int32_t ParseJson(uint32_t &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    if (!jsonValue.isUInt()) {
        CM_LOG_ERROR << ("Failed to convert the value of the " + key + " key to the [uint32] type.");
        return -1;
    }

    elem = static_cast<uint32_t>(jsonValue.asUInt());

    return 0;
}

int32_t ParseJson(std::string &elem, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    if (!jsonValue.isString()) {
        CM_LOG_ERROR << ("Failed to convert the value of the " + key + " key to the [string] type.");
        return -1;
    }

    elem = static_cast<std::string>(jsonValue.asString());

    return 0;
}

int32_t ParseJson(std::unordered_map<std::string, DataReaderQosInfo> &dataReaderInfoMap, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    for (Json::ArrayIndex i = 0; i < jsonValue.size(); ++i) {
        DataReaderQosInfo cmDataReaderQosInfo;
        ParseJson(cmDataReaderQosInfo.topic, jsonValue[i], "topic");
        ParseJson(cmDataReaderQosInfo.reliability, jsonValue[i], "reliability");
        ParseJson(cmDataReaderQosInfo.durability, jsonValue[i], "durability");
        ParseJson(cmDataReaderQosInfo.endpoint.history_memory_policy, jsonValue[i]["endpoint"], "history_memory_policy");
        ParseJson(cmDataReaderQosInfo.history.kind, jsonValue[i]["history"], "kind");
        ParseJson(cmDataReaderQosInfo.history.depth, jsonValue[i]["history"], "depth");
        ParseJson(cmDataReaderQosInfo.data_sharing, jsonValue[i], "data_sharing");

        dataReaderInfoMap.insert(std::make_pair(cmDataReaderQosInfo.topic, cmDataReaderQosInfo));
    }

    return 0;
}

int32_t ParseJson(std::unordered_map<std::string, DataWriterQosInfo> &dataWriterInfoMap, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    for (Json::ArrayIndex i = 0; i < jsonValue.size(); ++i) {
        DataWriterQosInfo cmDataWriterQosInfo;
        ParseJson(cmDataWriterQosInfo.topic, jsonValue[i], "topic");
        ParseJson(cmDataWriterQosInfo.reliability, jsonValue[i], "reliability");
        ParseJson(cmDataWriterQosInfo.durability, jsonValue[i], "durability");
        ParseJson(cmDataWriterQosInfo.endpoint.history_memory_policy, jsonValue[i]["endpoint"], "history_memory_policy");
        ParseJson(cmDataWriterQosInfo.history.kind, jsonValue[i]["history"], "kind");
        ParseJson(cmDataWriterQosInfo.history.depth, jsonValue[i]["history"], "depth");
        ParseJson(cmDataWriterQosInfo.data_sharing, jsonValue[i], "data_sharing");

        dataWriterInfoMap.insert(std::make_pair(cmDataWriterQosInfo.topic, cmDataWriterQosInfo));
    }

    return 0;
}

int32_t ParseJson(TransportQosInfo &transPortInfo, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    ParseJson(transPortInfo.use_builtin_transports, jsonValue, "use_builtin_transports");

    ParseJson(transPortInfo.udp_qos_info.enable, jsonValue["udp"], "enable");
    ParseJson(transPortInfo.udp_qos_info.network, jsonValue["udp"], "network");
    ParseJson(transPortInfo.udp_qos_info.send_socket_buffer_size, jsonValue["udp"], "send_socket_buffer_size");
    ParseJson(transPortInfo.udp_qos_info.listen_socket_buffer_size, jsonValue["udp"], "listen_socket_buffer_size");

    ParseJson(transPortInfo.shm_qos_info.enable, jsonValue["shm"], "enable");
    ParseJson(transPortInfo.shm_qos_info.max_message_size, jsonValue["shm"], "max_message_size");
    ParseJson(transPortInfo.shm_qos_info.segment_size, jsonValue["shm"], "segment_size");

    return 0;
}

int32_t ParseJson(DiscoveryQosInfo &discoverQosInfo, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    ParseJson(discoverQosInfo.typelookup_client, jsonValue, "typelookup_client");
    ParseJson(discoverQosInfo.typelookup_server, jsonValue, "typelookup_server");
    ParseJson(discoverQosInfo.leaseDuration, jsonValue, "leaseDuration");
    ParseJson(discoverQosInfo.leaseDuration_announce_period, jsonValue, "leaseDuration_announce_period");
    ParseJson(discoverQosInfo.initial_announce_count, jsonValue, "initial_announce_count");
    ParseJson(discoverQosInfo.initial_announce_period, jsonValue, "initial_announce_period");

    return 0;
}

int32_t ParseJson(std::unordered_map<uint32_t, ParticipantQosInfo> &participantparQosInfoMap, const Json::Value &currentNode, const std::string &key)
{
    Json::Value defaultValue;
    const auto &jsonValue = currentNode.get(key, defaultValue);
    if (jsonValue == defaultValue) {
        return -1;
    }

    for (Json::ArrayIndex i = 0; i < jsonValue.size(); ++i) {
        ParticipantQosInfo cmParticipantQosInfo;
        ParseJson(cmParticipantQosInfo.name, jsonValue[i], "name");
        ParseJson(cmParticipantQosInfo.domain_id, jsonValue[i], "domain");
        ParseJson(cmParticipantQosInfo.transport, jsonValue[i], "transport");
        ParseJson(cmParticipantQosInfo.discover, jsonValue[i], "discover");
        ParseJson(cmParticipantQosInfo.dataReaderInfo, jsonValue[i], "datareader");
        ParseJson(cmParticipantQosInfo.dataWriterInfo, jsonValue[i], "datawriter");

        participantparQosInfoMap.insert(std::make_pair(cmParticipantQosInfo.domain_id, cmParticipantQosInfo));
    }

    return 0;
}

int32_t CmQosConfig::ReadJsonFile(const std::string &filePath, std::unordered_map<uint32_t, ParticipantQosInfo> &participantparQosInfoMap)
{
    Json::Value root;
    int32_t ret = OpenJsonFile(filePath, root);
    if (ret < 0) {
        return -1;
    }

    ParseJson(participantparQosInfoMap, root, "participant");

    return 0;
}

void CmQosConfig::MappingParticipantQos(const ParticipantQosInfo& participantQosInfo, DomainParticipantQos& participantQos) {
    // transport qos config
    TransportQosInfo transportcfg = participantQosInfo.transport;
    participantQos.transport().use_builtin_transports = transportcfg.use_builtin_transports;

    if (transportcfg.udp_qos_info.enable == true) {
        auto udp_transport = std::make_shared<UDPv4TransportDescriptor>();
        udp_transport->interfaceWhiteList.emplace_back(transportcfg.udp_qos_info.network);
        participantQos.transport().send_socket_buffer_size = transportcfg.udp_qos_info.send_socket_buffer_size;
        participantQos.transport().listen_socket_buffer_size = transportcfg.udp_qos_info.listen_socket_buffer_size;
        participantQos.transport().user_transports.push_back(udp_transport);
    }

    if (transportcfg.shm_qos_info.enable == true) {
        auto shm_transport = std::make_shared<eprosima::fastdds::rtps::SharedMemTransportDescriptor>();
        shm_transport->segment_size(transportcfg.shm_qos_info.segment_size);
        shm_transport->max_message_size(transportcfg.shm_qos_info.max_message_size);
        participantQos.transport().user_transports.push_back(shm_transport);
    }


    // discover qos config
    DiscoveryQosInfo discovercfg = participantQosInfo.discover;
    participantQos.wire_protocol().builtin.typelookup_config.use_client = discovercfg.typelookup_client;
    participantQos.wire_protocol().builtin.typelookup_config.use_server = discovercfg.typelookup_server;

	participantQos.wire_protocol().builtin.discovery_config.leaseDuration = Duration_t(discovercfg.leaseDuration, 0);
    participantQos.wire_protocol().builtin.discovery_config.leaseDuration_announcementperiod = Duration_t(discovercfg.leaseDuration_announce_period, 0);
    participantQos.wire_protocol().builtin.discovery_config.initial_announcements.count = discovercfg.initial_announce_count;
    participantQos.wire_protocol().builtin.discovery_config.initial_announcements.period = Duration_t(0, (discovercfg.initial_announce_period * 1000 * 1000));

    participantQos.name(participantQosInfo.name);
}

void CmQosConfig::MappingDataWriterQos(const DataWriterQosInfo& dataWriterQosInfo, DataWriterQos& dataWriterQos) {

    switch(cm_qos_reliability[dataWriterQosInfo.reliability]) {
        case CM_RELIABILITY_RELIABLE:
            dataWriterQos.reliability().kind = RELIABLE_RELIABILITY_QOS;
            break;
        case CM_RELIABILITY_BEST_EFFORT:
            dataWriterQos.reliability().kind = BEST_EFFORT_RELIABILITY_QOS;
            break;
        default:
            dataWriterQos.reliability().kind = RELIABLE_RELIABILITY_QOS;
            break;
    }

    switch(cm_qos_durability[dataWriterQosInfo.durability]) {
        case CM_DURABILITY_VOLATILE:
            dataWriterQos.durability().kind = VOLATILE_DURABILITY_QOS;
            break;
        case CM_DURABILITY_TRANSIENT_LOCAL:
            dataWriterQos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;
            break;
        case CM_DURABILITY_TRANSIENT:
            dataWriterQos.durability().kind = TRANSIENT_DURABILITY_QOS;
            break;
        default:
            dataWriterQos.durability().kind = VOLATILE_DURABILITY_QOS;
            break;
    }

    switch(cm_qos_history_kind[dataWriterQosInfo.history.kind]) {
        case CM_HISTORY_KEEP_LAST:
            dataWriterQos.history().kind = KEEP_LAST_HISTORY_QOS;
            dataWriterQos.history().depth =  dataWriterQosInfo.history.depth;
            break;
        case CM_HISTORY_KEEP_ALL:
            dataWriterQos.history().kind = KEEP_ALL_HISTORY_QOS;
            dataWriterQos.history().depth =  dataWriterQosInfo.history.depth;
            break;
        default:
            dataWriterQos.history().kind = KEEP_LAST_HISTORY_QOS;
            dataWriterQos.history().depth =  dataWriterQosInfo.history.depth;
            break;
    }

    switch(cm_qos_data_sharing[dataWriterQosInfo.data_sharing]) {
        case CM_DATA_SHARE_OFF:
            dataWriterQos.data_sharing().off();
            break;
        case CM_DATA_SHARE_ON:
            dataWriterQos.data_sharing().on(".");
            break;
        case CM_DATA_SHARE_AUTO:
            dataWriterQos.data_sharing().automatic();
            break;
        default:
            dataWriterQos.data_sharing().off();
            break;
    }

    switch(cm_qos_memory_policy[dataWriterQosInfo.endpoint.history_memory_policy]) {
        case CM_PREALLOCATED:
            dataWriterQos.endpoint().history_memory_policy = PREALLOCATED_MEMORY_MODE;
            break;
        case CM_PREALLOCATED_WITH_REALLOC:
            dataWriterQos.endpoint().history_memory_policy = PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
            break;
        case CM_DYNAMIC_RESERVE:
            dataWriterQos.endpoint().history_memory_policy = DYNAMIC_RESERVE_MEMORY_MODE;
            break;
        case CM_DYNAMIC_REUSABLE:
            dataWriterQos.endpoint().history_memory_policy = DYNAMIC_REUSABLE_MEMORY_MODE;
            break;
        default:
            dataWriterQos.endpoint().history_memory_policy = PREALLOCATED_MEMORY_MODE;
            break;
    }
}

void CmQosConfig::MappingDataReaderQos(const DataReaderQosInfo& dataReaderQosInfo, DataReaderQos& dataReaderQos) {

    switch(cm_qos_reliability[dataReaderQosInfo.reliability]) {
        case CM_RELIABILITY_RELIABLE:
            dataReaderQos.reliability().kind = RELIABLE_RELIABILITY_QOS;
            break;
        case CM_RELIABILITY_BEST_EFFORT:
            dataReaderQos.reliability().kind = BEST_EFFORT_RELIABILITY_QOS;
            break;
        default:
            dataReaderQos.reliability().kind = RELIABLE_RELIABILITY_QOS;
            break;
    }

    switch(cm_qos_durability[dataReaderQosInfo.durability]) {
        case CM_DURABILITY_VOLATILE:
            dataReaderQos.durability().kind = VOLATILE_DURABILITY_QOS;
            break;
        case CM_DURABILITY_TRANSIENT_LOCAL:
            dataReaderQos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;
            break;
        default:
            dataReaderQos.durability().kind = VOLATILE_DURABILITY_QOS;
            break;
    }

    switch(cm_qos_history_kind[dataReaderQosInfo.history.kind]) {
        case CM_HISTORY_KEEP_LAST:
            dataReaderQos.history().kind = KEEP_LAST_HISTORY_QOS;
            dataReaderQos.history().depth =  dataReaderQosInfo.history.depth;
            break;
        case CM_HISTORY_KEEP_ALL:
            dataReaderQos.history().kind = KEEP_ALL_HISTORY_QOS;
            dataReaderQos.history().depth =  dataReaderQosInfo.history.depth;
            break;
        default:
            dataReaderQos.history().kind = KEEP_LAST_HISTORY_QOS;
            dataReaderQos.history().depth =  dataReaderQosInfo.history.depth;
            break;
    }

    switch(cm_qos_data_sharing[dataReaderQosInfo.data_sharing]) {
        case CM_DATA_SHARE_OFF:
            dataReaderQos.data_sharing().off();
            break;
        case CM_DATA_SHARE_ON:
            dataReaderQos.data_sharing().on(".");
            break;
        case CM_DATA_SHARE_AUTO:
            dataReaderQos.data_sharing().automatic();
            break;
        default:
            dataReaderQos.data_sharing().off();
            break;
    }

    switch(cm_qos_memory_policy[dataReaderQosInfo.endpoint.history_memory_policy]) {
        case CM_PREALLOCATED:
            dataReaderQos.endpoint().history_memory_policy = PREALLOCATED_MEMORY_MODE;
            break;
        case CM_PREALLOCATED_WITH_REALLOC:
            dataReaderQos.endpoint().history_memory_policy = PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
            break;
        case CM_DYNAMIC_RESERVE:
            dataReaderQos.endpoint().history_memory_policy = DYNAMIC_RESERVE_MEMORY_MODE;
            break;
        case CM_DYNAMIC_REUSABLE:
            dataReaderQos.endpoint().history_memory_policy = DYNAMIC_REUSABLE_MEMORY_MODE;
            break;
        default:
            dataReaderQos.endpoint().history_memory_policy = PREALLOCATED_MEMORY_MODE;
            break;
    }
}

int32_t CmQosConfig::GetNetworkList(std::vector<std::string>& network_list) {
    Json::Value root;

    char* filePath = getenv("CONF_PREFIX_PATH");
    if (filePath == NULL) {
        return -1;
    }
    std::string env_path = filePath;

    int32_t ret = OpenJsonFile(env_path + "default_network_list.json", root);
    if (ret < 0) {
        return ret;
    }
    ParseJson(network_list, root, "network_list");
    if (0 == network_list.size()) {
        return -1;
    }
    return 0;
}

DomainParticipantQos CmQosConfig::GetParticipantQos(const uint32_t domain) {
    DomainParticipantQos particpant_qos = PARTICIPANT_QOS_DEFAULT;
    std::lock_guard<std::mutex> lk(_mutex);

    if (_qos_config_map.find(domain) == _qos_config_map.end()) {
        CM_LOG_WARN << "[domain: " << domain << "] use default qos profile.";
        return particpant_qos;
    }

    MappingParticipantQos(_qos_config_map[domain], particpant_qos);

    ParticipantQosInfo p_qos = _qos_config_map[domain];
    CM_LOG_INFO << "[domain: " << domain << "] participant qos profile "
        << " use_builtin_transports : " << p_qos.transport.use_builtin_transports
        << " udp.enable : " << p_qos.transport.udp_qos_info.enable
        << " udp.network : " << p_qos.transport.udp_qos_info.network
        << " udp.send_socket_buffer : " << p_qos.transport.udp_qos_info.send_socket_buffer_size
        << " udp.listen_socket_buffer : " << p_qos.transport.udp_qos_info.listen_socket_buffer_size
        << " shm.enable : " << p_qos.transport.shm_qos_info.enable
        << " shm.max_message_size : " << p_qos.transport.shm_qos_info.max_message_size
        << " shm.segment_size : " << p_qos.transport.shm_qos_info.segment_size;

    CM_LOG_INFO << "[domain: " << domain << "] discovery qos profile " 
        << " initial_announce_count : " << p_qos.discover.initial_announce_count
        << " initial_announce_period : " << p_qos.discover.initial_announce_period
        << " leaseDuration : " << p_qos.discover.leaseDuration
        << " leaseDuration_announce_period : " << p_qos.discover.leaseDuration_announce_period
        << " typelookup_client : " << p_qos.discover.typelookup_client
        << " typelookup_server : " << p_qos.discover.typelookup_server;

    return particpant_qos;
}

DataWriterQos CmQosConfig::GetWriterQos(const uint32_t domain, const std::string& topic) {
    DataWriterQos writer_qos = DATAWRITER_QOS_DEFAULT;
    std::lock_guard<std::mutex> lk(_mutex);

    if (_qos_config_map.find(domain) == _qos_config_map.end()) {
        CM_LOG_ERROR << "[domain: " << domain << " topic: " << topic << "] not found domain writer qos profile.";
        return writer_qos;
    }

    ParticipantQosInfo& p_qos = _qos_config_map[domain];
    if (p_qos.dataWriterInfo.find(topic) != p_qos.dataWriterInfo.end()) {
        MappingDataWriterQos(p_qos.dataWriterInfo[topic], writer_qos);
    } else {
        if (p_qos.dataWriterInfo.find(DEFAULT_QOS_TOPIC) != p_qos.dataWriterInfo.end()) {
            CM_LOG_WARN << "[domain: " << domain << " topic: " << topic << "] use " << DEFAULT_QOS_TOPIC <<" datawriter qos profile.";
            MappingDataWriterQos(p_qos.dataWriterInfo[DEFAULT_QOS_TOPIC], writer_qos);
        } else {
            CM_LOG_ERROR << "[domain: " << domain << " topic: " << topic << "] nod found " << DEFAULT_QOS_TOPIC << " writer profile.";
            return writer_qos;
        }
    }

    return writer_qos;
}

DataReaderQos CmQosConfig::GetReaderQos(const uint32_t domain, const std::string& topic) {
    DataReaderQos reader_qos = DATAREADER_QOS_DEFAULT;
    std::lock_guard<std::mutex> lk(_mutex);

    if (_qos_config_map.find(domain) == _qos_config_map.end()) {
        CM_LOG_ERROR << "[domain: " << domain << " topic: " << topic << "] not found domain reader qos profile.";
        return reader_qos;
    }

    ParticipantQosInfo& p_qos = _qos_config_map[domain];
    if (p_qos.dataReaderInfo.find(topic) != p_qos.dataReaderInfo.end()) {
        MappingDataReaderQos(p_qos.dataReaderInfo[topic], reader_qos);
    } else {
        if (p_qos.dataReaderInfo.find(DEFAULT_QOS_TOPIC) != p_qos.dataReaderInfo.end()) {
            CM_LOG_WARN << "[domain: " << domain << " topic: " << topic << "] use " << DEFAULT_QOS_TOPIC << " datareader qos profile.";
            MappingDataReaderQos(p_qos.dataReaderInfo[DEFAULT_QOS_TOPIC], reader_qos);
        } else {
            CM_LOG_ERROR << "[domain: " << domain << " topic: " << topic << "] nod found reader " << DEFAULT_QOS_TOPIC << " qos profile.";
            return reader_qos;
        }
    }

    return reader_qos;
}

DataReaderQos CmQosConfig::GetReaderQos(const uint32_t domain, const std::string& topic, QosMode mode) {
    DataReaderQos reader_qos = DATAREADER_QOS_DEFAULT;
    std::lock_guard<std::mutex> lk(_mutex);

    if (_qos_config_map.find(domain) == _qos_config_map.end()) {
        CM_LOG_ERROR << "[domain: " << domain << " topic: " << topic << "] not found domain reader qos profile.";
        return reader_qos;
    }

    ParticipantQosInfo& p_qos = _qos_config_map[domain];

    if (p_qos.dataReaderInfo.find(topic) != p_qos.dataReaderInfo.end()) {
        MappingDataReaderQos(p_qos.dataReaderInfo[topic], reader_qos);
    } else {
        if (p_qos.dataReaderInfo.find(DEFAULT_QOS_TOPIC) != p_qos.dataReaderInfo.end()) {
            CM_LOG_WARN << "[domain: " << domain << " topic: " << topic << "] use " << DEFAULT_QOS_TOPIC << " datareader qos profile.";
            MappingDataReaderQos(p_qos.dataReaderInfo[DEFAULT_QOS_TOPIC], reader_qos);
        } else {
            CM_LOG_ERROR << "[domain: " << domain << " topic: " << topic << "] nod found reader " << DEFAULT_QOS_TOPIC << " qos profile.";
            return reader_qos;
        }
    }

    return reader_qos;
}

int32_t CmQosConfig::GetQosFromEnv(const std::string env, const uint32_t domain) {
    std::string qos_file;
    char* fileName = getenv(env.c_str());
    if (fileName == NULL) {
#ifdef BUILD_FOR_ORIN
        qos_file = "/app/conf/default_qos.json";
#else
        char* default_path = getenv(DEFAULT_CONFIG_PATH);
        if (default_path == NULL) {
            CM_LOG_ERROR << "[domain: " << domain << "] not found default conf path.";
            return -1;
        } else {
            qos_file = default_path + std::string("default_qos.json");
        }
#endif
        CM_LOG_ERROR << "[domain: " << domain << "] not found etc qos json, use : " << qos_file;
    } else {
        qos_file = fileName;
    }

    {
        std::lock_guard<std::mutex> lk(_mutex);
        if (_qos_config_map.find(domain) != _qos_config_map.end()) {
            CM_LOG_INFO << "[domain: " << domain << "] use parse qos path [" << qos_file << "]";
            return 0;
        }

        int ret = ReadJsonFile(qos_file, _qos_config_map);
        if (ret < 0) {
            CM_LOG_ERROR << "[domain: " << domain << "] qos path [ " << qos_file << " ] read failed.";
            return -1;
        }
    }

    CM_LOG_INFO << "[domain: " << domain << "] qos path [" << qos_file << "] read success.";

    return 0;
}

int32_t CmQosConfig::GetQosFromEnv(const std::string env, const uint32_t domain, const std::string file_name) {
    std::string qos_file;
    char* filePath = getenv(env.c_str());
    if (filePath == NULL) {
#ifdef BUILD_FOR_ORIN
        qos_file = "/app/conf/default_qos.json";
#else
        char* default_path = getenv(DEFAULT_CONFIG_PATH);
        if (default_path == NULL) {
            CM_LOG_ERROR << "[domain: " << domain << "] not found default conf path.";
            return -1;
        } else {
            qos_file = default_path + std::string("default_qos.json");
        }
#endif
        CM_LOG_ERROR << "[domain: " << domain << "] not found conf qos json, use : " << qos_file;
    } else {
        qos_file = filePath + file_name;
    }

    {
        std::lock_guard<std::mutex> lk(_mutex);
        if (_qos_config_map.find(domain) != _qos_config_map.end()) {
            CM_LOG_INFO << "[domain: " << domain << "] use parse qos path [" << qos_file << "]";
            return 0;
        }

        int ret = ReadJsonFile(qos_file, _qos_config_map);
        if (ret < 0) {
            CM_LOG_ERROR << "[domain: " << domain << "] qos path [ " << qos_file << " ] read failed.";
            return -1;
        }
    }

    CM_LOG_INFO << "[domain: " << domain << "] qos path [" << qos_file << "] read success.";

    return 0;
}

}
}
}