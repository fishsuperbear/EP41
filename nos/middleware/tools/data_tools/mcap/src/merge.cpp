#include <fstream>
#include <sstream>
#include "merge.h"
#include "rosbag2_cpp/writer.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "mcap_common.hpp"
#include "reader.h"

namespace hozon {
namespace netaos {
namespace mcap {

Merge::Merge() {}

Merge::~Merge() {}

void Merge::merge_mcap(std::vector<std::string> mcap_file_path_vec,
                       std::vector<std::string> attachment_str_name_vec,
                       std::vector<std::string> attachment_str_vec,
                       std::vector<std::string> attachment_file_path_vec,
                       std::string output_file_path) {
    m_output_file_path_vec.clear();
    std::unique_ptr<rosbag2_cpp::Writer> writer;
    writer = std::make_unique<rosbag2_cpp::Writer>();
    
    //将所有非mcap类型的数据，合入mcap
    for (uint i = 0; i < attachment_str_name_vec.size(); i++) {
        std::shared_ptr<rosbag2_storage::Attachment> attachment = std::make_shared<rosbag2_storage::Attachment>();
        struct timespec time_now;
        clock_gettime(CLOCK_REALTIME, &time_now);
        attachment->logTime = static_cast<double>(time_now.tv_sec) + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
        attachment->createTime = 0;
        attachment->name = attachment_str_name_vec[i];
        attachment->mediaType = attachment_str_name_vec[i];
        attachment->data = attachment_str_vec[i];
        attachment->dataSize = attachment_str_vec[i].length();
        writer->add_attach(attachment);
    }

    //将所有非mcap类型的文件，合入mcap
    for (std::string attachment_file_path : attachment_file_path_vec) {
        std::shared_ptr<rosbag2_storage::Attachment> attachment = std::make_shared<rosbag2_storage::Attachment>();
        std::ifstream file_stream(attachment_file_path, std::ios::in | std::ios::binary);
        std::stringstream file_sstr;
        file_sstr << file_stream.rdbuf();
        file_stream.close();
        struct timespec time_now;
        clock_gettime(CLOCK_REALTIME, &time_now);
        attachment->logTime = static_cast<double>(time_now.tv_sec) + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
        attachment->createTime = 0;
        attachment->name = attachment_file_path;
        size_t dotPosition = attachment_file_path.find_last_of(".");
        attachment->mediaType = (dotPosition != std::string::npos && dotPosition < attachment_file_path.length() - 1) ? attachment_file_path.substr(dotPosition + 1) : "";
        attachment->data = file_sstr.str();
        attachment->dataSize = file_sstr.str().length();
        writer->add_attach(attachment);
    }

    //将所有mcap类型的文件的原本附件，合入新mcap
    for (auto mcap_file_path : mcap_file_path_vec) {
        std::unique_ptr<rosbag2_cpp::Reader> reader;
        reader = std::make_unique<rosbag2_cpp::Reader>();
        reader->open(mcap_file_path);
        std::vector<std::string> attachment_vec;
        reader->get_all_attachments_filepath(attachment_vec);
        for (std::string attachment : attachment_vec) {
            std::shared_ptr<rosbag2_storage::Attachment> attachment_ptr = reader->read_Attachment(attachment);
            if (attachment_ptr != nullptr) {
                writer->add_attach(attachment_ptr);
            }
        }
        reader->close();
    }

    //将所有mcap类型的文件，合入新mcap
    rosbag2_cpp::bag_events::WriterEventCallbacks callback;
    callback.write_split_callback = std::bind(&Merge::set_output_file_path_vec, this, std::placeholders::_1);
    writer->add_event_callbacks(callback);
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = output_file_path;
    rosbag2_cpp::ConverterOptions converter_options{};
    writer->open(storage_options, converter_options);
    for (auto mcap_file_path : mcap_file_path_vec) {
        std::unique_ptr<rosbag2_cpp::Reader> reader;
        reader = std::make_unique<rosbag2_cpp::Reader>();
        reader->open(mcap_file_path);
        while (reader->has_next()) {
            std::shared_ptr<rosbag2_storage::SerializedBagMessage> message_ptr = reader->read_next();
            std::string topic_name = message_ptr->topic_name;
            std::string type_name;
            std::vector<rosbag2_storage::TopicMetadata> topic_meta_data = reader->get_all_topics_and_types();
            for (auto temp : topic_meta_data) {
                if (temp.name == message_ptr->topic_name) {
                    type_name = temp.type;
                }
            }
            writer->write(message_ptr, topic_name, type_name, "cdr");
        }
        reader->close();
    }
}

void Merge::merge_mcap(std::vector<std::string> mcap_file_path_vec,
                       std::vector<std::string> attachment_file_path_vec,
                       std::string output_file_path) {
    std::vector<std::string> attachment_str_name_vec;
    std::vector<std::string> attachment_str_vec;
    merge_mcap(mcap_file_path_vec, attachment_str_name_vec, attachment_str_vec, attachment_file_path_vec, output_file_path);
}

void Merge::merge_mcap(std::vector<std::string> mcap_file_path_vec,
                       std::vector<std::string> attachment_str_name_vec,
                       std::vector<std::string> attachment_str_vec,
                       std::string output_file_path) {
    std::vector<std::string> attachment_file_path_vec;
    merge_mcap(mcap_file_path_vec, attachment_str_name_vec, attachment_str_vec, attachment_file_path_vec, output_file_path);
}

void Merge::merge_mcap(std::vector<std::string> mcap_file_path_vec,
                       std::string output_file_path) {
    std::vector<std::string> attachment_str_name_vec;
    std::vector<std::string> attachment_str_vec;
    std::vector<std::string> attachment_file_path_vec;
    merge_mcap(mcap_file_path_vec, attachment_str_name_vec, attachment_str_vec, attachment_file_path_vec, output_file_path);
}

void Merge::set_output_file_path_vec(rosbag2_cpp::bag_events::BagSplitInfo& info) {
    std::string filePath;
    if (info.opened_file.empty()) {
        filePath = info.closed_file;
    } else {
        filePath = info.opened_file;
    }
    const std::string suffix = ".active";
    size_t suffixIndex = filePath.rfind(suffix);
    if (suffixIndex==filePath.size()-suffix.size()) {
        filePath = filePath.substr(0, suffixIndex);
    }
    m_output_file_path_vec.insert(filePath);
}

std::set<std::string> Merge::get_output_file_path_vec() {
    return m_output_file_path_vec;
}

}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
