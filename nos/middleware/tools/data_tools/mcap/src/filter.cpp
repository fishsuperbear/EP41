#include <algorithm>
#include <fstream>
#include <sstream>
#include <regex>
#include "filter.h"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_cpp/writer.hpp"
#include "mcap_common.hpp"
#include "data_tools_logger.hpp"

namespace hozon {
namespace netaos {
namespace mcap {

Filter::Filter() {}

Filter::~Filter() {}

bool Filter::regular_matching_topic(std::vector<std::string> regular_topic_vec, std::string topic) {
    for (auto regular_topic : regular_topic_vec) {
        std::regex regular_text(regular_topic);
        if (std::regex_match(topic, regular_text)) {
            return true;
        }
    }
    return false;
}

void Filter::filter_base_topic_list(std::string mcap_file_path,
                                    std::vector<std::string> regular_white_topic_vec,
                                    std::vector<std::string> regular_black_topic_vec,
                                    std::string output_folder_path) {
    std::ifstream mcap_file_if(mcap_file_path, std::ios::in | std::ios::binary);
    if (!mcap_file_if) {
        BAG_LOG_WARN << mcap_file_path << " file open failed";
        return;
    }
    mcap_file_if.seekg(0, std::ios::end);
    long long mcap_file_size = mcap_file_if.tellg();
    mcap_file_if.close();
    if (mcap_file_size == 0) {
        BAG_LOG_WARN << mcap_file_path << " file is empty";
        return;
    }

    std::string mcap_file_name = GetFileName(mcap_file_path);
    m_output_file_path.clear();
    std::unique_ptr<rosbag2_cpp::Writer> writer;
    writer = std::make_unique<rosbag2_cpp::Writer>();
    rosbag2_cpp::bag_events::WriterEventCallbacks callback;
    callback.write_split_callback = std::bind(&Filter::set_output_file_path, this, std::placeholders::_1);
    writer->add_event_callbacks(callback);
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = output_folder_path + "/" + mcap_file_name;
    storage_options.use_time_suffix = false;
    rosbag2_cpp::ConverterOptions converter_options{};
    writer->open(storage_options, converter_options);

    std::unique_ptr<rosbag2_cpp::Reader> reader;
    reader = std::make_unique<rosbag2_cpp::Reader>();
    reader->open(mcap_file_path);
    while (reader->has_next()) {
        std::shared_ptr<rosbag2_storage::SerializedBagMessage> message_ptr = reader->read_next();
        std::string topic = message_ptr->topic_name;
        std::string type;
        std::vector<rosbag2_storage::TopicMetadata> topic_meta_data = reader->get_all_topics_and_types();
        for (auto temp : topic_meta_data) {
            if (temp.name == message_ptr->topic_name) {
                type = temp.type;
            }
        }
        int is_write = 0;
        if (regular_white_topic_vec.empty() == false) {
            if (regular_matching_topic(regular_white_topic_vec, topic) == true) {
                is_write++;
            }
        } else {
            is_write++;
        }
        if (regular_black_topic_vec.empty() == false) {
            if (regular_matching_topic(regular_black_topic_vec, topic) == false) {
                is_write++;
            }
        } else {
            is_write++;
        }
        if (is_write == 2) {
            writer->write(message_ptr, topic, type, "cdr");
        }
    }
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

void Filter::set_output_file_path(rosbag2_cpp::bag_events::BagSplitInfo& info) {
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
    m_output_file_path = filePath;
}

std::string Filter::get_output_file_path() {
    return m_output_file_path;
}


}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
