#include "adf/include/proxy/node_proxy_cm.h"
#include "adf/include/base.h"
#include "adf/include/internal_log.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeProxyCM::NodeProxyCM(const NodeConfig::CommInstanceConfig& config,
                         std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type)
    : NodeProxyBase(config), _domain(config.domain) {
    _topic = config.topic;
    // if(data != nullptr) {
    //     std::shared_ptr<eprosima::fastdds::dds::TopicDataType>
    //          pub_sub_type = *(reinterpret_cast<std::shared_ptr<eprosima::fastdds::dds::TopicDataType>*>(data));
    _pub_sub_type = pub_sub_type;
    PROXY_INIT(_proxy, _pub_sub_type, _domain, _topic, NodeProxyCM::OnDataReceive);
    _freq_monitor.Start();
}

NodeProxyCM::~NodeProxyCM() {
    _freq_monitor.Stop();
}

BaseDataTypePtr NodeProxyCM::CreateBaseDataFromIDL(std::shared_ptr<IDLBaseType> idl_msg) {
    BaseDataTypePtr alg_data = std::make_shared<BaseData>();
    alg_data->idl_msg = idl_msg;
    alg_data->__header.seq = idl_msg->header().seq();
    alg_data->__header.timestamp_virt_us =
        TimestampToUs(idl_msg->header().timestamp_virt().sec(), idl_msg->header().timestamp_virt().nsec());
    alg_data->__header.timestamp_real_us =
        TimestampToUs(idl_msg->header().timestamp_real().sec(), idl_msg->header().timestamp_real().nsec());

    for (auto linkinfo : idl_msg->header().latency_info().link_infos()) {
        alg_data->__header.latency_info.data[linkinfo.link_name()].sec = linkinfo.timestamp_real().sec();
        alg_data->__header.latency_info.data[linkinfo.link_name()].nsec = linkinfo.timestamp_real().nsec();
    }

    return alg_data;
}

void NodeProxyCM::OnDataReceive(void) {
    std::shared_ptr<IDLBaseType> idl_msg(static_cast<IDLBaseType*>(_pub_sub_type->createData()));

    _proxy->Take(idl_msg);
    ADF_LOG_TRACE << "Proxy receive " << _config.name;

    // cm data conver to alg data
    BaseDataTypePtr alg_data = CreateBaseDataFromIDL(idl_msg);

    PushOneAndNotify(alg_data);
    _freq_monitor.PushOnce();
}

void NodeProxyCM::PauseReceive() {
    // PROXY_DEINIT(_proxy)
}

void NodeProxyCM::ResumeReceive() {
    // PROXY_INIT(_proxy, _pub_sub_type, _domain, _topic, NodeProxyCM::OnDataReceive)
}

void NodeProxyCM::Deinit() {
    ADF_LOG_DEBUG << "CM Proxy deinit.";
    PROXY_DEINIT(_proxy)
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
