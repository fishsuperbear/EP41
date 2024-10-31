/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: bag_record.cpp
 * @Date: 2023/07/13
 * @Author: cheng
 * @Desc: --
 */


#include "collection/include/impl/bag_record.h"
#include "utils/include/path_utils.h"
#include "middleware/idl/generated/cm_protobuf.h"
#include "middleware/idl/generated/cm_protobufPubSubTypes.h"
#include "proto/dead_reckoning/dr.pb.h"

namespace hozon {
namespace netaos {
namespace dc {

void BagRecorder::active() {
    {
        std::filesystem::path needCleanPath(rops_.output_file_name);
        std::string parentPath = needCleanPath.parent_path();
        if (parentPath.size()>10 && delFileseBeforeRec_) {
            PathUtils::removeFilesInFolder(parentPath);
        }
        rec_->RegisterPreWriteCallbak("/localization/deadreckoning", std::bind(&BagRecorder::clearWGS, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        rec_->Start(rops_);
        status_.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    }
}

void BagRecorder::configure(std::string type, YAML::Node &node) {
    if (type=="delFileseBeforeRec") {
        delFileseBeforeRec_ = node["delFileseBeforeRec"].as<bool>();
        return;
    }
    if (type=="video") {
        filesType_ = FilesInListType::videoTopicMcapFiles;
    } else  {
        filesType_ = FilesInListType::commonTopicMcapFiles;
    }
    rops_ = node.as<bag::RecordOptions>();
    status_.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void BagRecorder::configure(std::string type,  DataTrans &dataTrans){

}

void BagRecorder::deactive() {
    std::cout << "=======" << __FILE__ << ":" << __LINE__ << "\n";
    rec_->Stop();
    std::cout << "=======" << __FILE__ << ":" << __LINE__ << "\n";
    status_.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
}
TaskStatus BagRecorder::getStatus() {
    return status_.load(std::memory_order::memory_order_acquire);
}

bag::RecordErrorCode BagRecorder::clearWGS(bag::BagMessage& cur_msg, std::vector<bag::BagMessage>& pre_msg, std::vector<bag::BagMessage>& post_msg) {
    hozon::dead_reckoning::DeadReckoning protoData;
    std::string protoName = protoData.GetTypeName();
    CmProtoBuf cmProtoBuf;
    cmProtoBuf.name(protoName);
    CmProtoBufPubSubType sub_type;
    sub_type.deserialize(cur_msg.data.m_payload.get(), &cmProtoBuf);
    protoData.ParseFromArray(cmProtoBuf.str().data(), cmProtoBuf.str().size());
    if (protoData.has_pose() && protoData.pose().has_pose_wgs()) {
        protoData.mutable_pose()->clear_pose_wgs();
    }
    std::string serializedStr;
    protoData.SerializeToString(&serializedStr);
    cmProtoBuf.str().clear();
    cmProtoBuf.str().assign(serializedStr.begin(), serializedStr.end());
    sub_type.serialize(&cmProtoBuf, cur_msg.data.m_payload.get());
    return bag::RecordErrorCode::SUCCESS;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
