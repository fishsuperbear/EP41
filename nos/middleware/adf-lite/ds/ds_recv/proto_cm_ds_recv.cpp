#include "adf-lite/ds/ds_recv/proto_cm_ds_recv.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "adf/include/node_proto_register.h"


namespace hozon {
namespace netaos {
namespace adf_lite {
    
ProtoCMDsRecv::ProtoCMDsRecv(const DSConfig::DataSource& config) :
        DsRecv(config),
        _initialized(false),
        _proxy(std::make_shared<CmProtoBufPubSubType>()) {
    ResumeReceive();
    _writer.Init(_config.topic);
}

ProtoCMDsRecv::~ProtoCMDsRecv() {
}

void ProtoCMDsRecv::Deinit() {
    PauseReceive();
}

void ProtoCMDsRecv::OnDataReceive(void) {
    std::shared_ptr<CmProtoBuf> cm_data(new CmProtoBuf);
    if (cm_data == nullptr) {
        DS_LOG_ERROR << "cm_data pointer is nullptr!!!";
    }
    int32_t ret = _proxy.Take(cm_data);
    if (ret < 0) {
        DS_LOG_ERROR << "Fail to take cm data of topic " << _config.topic;
        return;
    }

    std::shared_ptr<google::protobuf::Message> msg = adf::ProtoMessageTypeMgr::GetInstance().Create(_config.topic);
    if (!msg) {
        DS_LOG_ERROR << "Fail to find topic prototype " << _config.topic;
        return;
    }

    bool bret = msg->ParseFromArray(cm_data->str().data(), cm_data->str().size());
    if (!bret) {
        DS_LOG_ERROR << "Fail to parse proto " << _config.topic;
        return;
    }

    BaseDataTypePtr base_ptr(new BaseData);
    if (base_ptr == nullptr) {
        DS_LOG_ERROR << "base_ptr pointer is nullptr!!!";
    }

    DS_LOG_VERBOSE << "check header info: data->__header.timestamp_real_us = " << base_ptr->__header.timestamp_real_us;
    base_ptr->__header.timestamp_real_us = 0;
    // 将cm_data中的header信息复制到base_ptr中
    if (cm_data->header().latency_info().link_infos().size() > 0) {

        for (auto linkinfo : cm_data->header().latency_info().link_infos()) {
            DS_LOG_DEBUG << "base_ptr->__header.latency_info().link_infos():"
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].sec << " "
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].nsec;
            base_ptr->__header.latency_info.data[linkinfo.link_name()].sec = linkinfo.timestamp_real().sec();
            base_ptr->__header.latency_info.data[linkinfo.link_name()].nsec = linkinfo.timestamp_real().nsec();
            DS_LOG_DEBUG << "base_ptr->__header.latency_info().link_infos():"
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].sec << " "
                         << base_ptr->__header.latency_info.data[linkinfo.link_name()].nsec;
        }
    } else {
        DS_LOG_DEBUG << "_config.topic:" << _config.topic << " cm_data->header has not link_infos, size is: "
                     << cm_data->header().latency_info().link_infos().size();
    }

    base_ptr->proto_msg = msg;
    ret = _writer.Write(base_ptr);
    if (ret < 0) {
        DS_LOG_ERROR << "Fail to write " << _config.topic;
        return;
    }

    DS_LOG_DEBUG << "Recv proto data from cm " << _config.cm_topic << ", Send lite topic: " << _config.topic;
}

void ProtoCMDsRecv::PauseReceive() {
    if (_initialized) {
        bool expected = true;
        bool newValue = false;
        bool result = _initialized.compare_exchange_weak(expected, newValue);
        if (result) {
            DS_LOG_INFO << "PauseReceive: " << _config.cm_topic;
            _proxy.Deinit();
        }
    } else {
        DS_LOG_INFO << "PauseReceive do nothing because _initialized value has been false";
    }
}

void ProtoCMDsRecv::ResumeReceive() {
    if (!_initialized) {
        bool expected = false;
        bool newValue = true;
        bool result = _initialized.compare_exchange_weak(expected, newValue);
        if (result) {
            DS_LOG_INFO << "ResumeReceive: " << _config.cm_topic << " domain_id:" << _config.cm_domain_id;
            _proxy.Init(_config.cm_domain_id, _config.cm_topic);
            _proxy.Listen(std::bind(&ProtoCMDsRecv::OnDataReceive, this));
        } else {
            DS_LOG_INFO << "ResumeReceive do nothing because _initialized value has been true";
        }
    }
}

}
}
}

