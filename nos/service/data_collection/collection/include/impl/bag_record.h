/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: bag_record.h
 * @Date: 2023/07/13
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_IMPL_BAG_RECORD_H_
#define MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_IMPL_BAG_RECORD_H_

#include <filesystem>
#include <signal.h>

#include "basic/trans_struct.h"
#include "collection/include/collection.h"
#include "dc_macros.h"
#include "include/yaml_cpp_struct.hpp"
#include "recorder.h"
#include "utils/include/dc_logger.hpp"
#include "fastdds/rtps/common/SerializedPayload.h"
#include <memory>

YCS_ADD_STRUCT(hozon::netaos::bag::RecordOptions, storage_id, max_bagfile_size, max_bagfile_duration,
               record_all, topics, output_file_name, max_files)
namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos;
class BagRecorder : public Collection {
public:
 BagRecorder() {
        DC_SERVER_LOG_DEBUG<<"for debug";
        rec_ = std::make_shared<bag::Recorder>();
        std::cout<<"============="<<__FILE__<<":"<<__LINE__<<std::endl;
//        DC_SERVER_LOG_DEBUG<<"for debug";
    }
    ~BagRecorder() override {
    };
    void onCondition(std::string type, char *data, Callback callback) override {}
    void configure(std::string type, YAML::Node &node) override;
    void configure(std::string type, DataTrans &node);
    void active() override;
    void deactive() override;
    TaskStatus getStatus() override;
    bool getTaskResult(const std::string &type, struct DataTrans & dataStruct) override {
        dataStruct.dataType = DataTransType::folder;
        std::filesystem::path fullPath(rops_.output_file_name);
        dataStruct.pathsList[filesType_].insert(fullPath.parent_path().string());
        return true;
    };
    void pause() override {}
    bag::RecordErrorCode clearWGS(bag::BagMessage& cur_msg, std::vector<bag::BagMessage>& pre_msg, std::vector<bag::BagMessage>& post_msg);
protected:
    std::shared_ptr<bag::Recorder> rec_{nullptr};
    bag::RecordOptions rops_;
    std::atomic_bool delFileseBeforeRec_{false};
    FilesInListType filesType_;
    std::atomic<TaskStatus> status_{TaskStatus::INITIAL};
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_IMPL_BAG_RECORD_H_
