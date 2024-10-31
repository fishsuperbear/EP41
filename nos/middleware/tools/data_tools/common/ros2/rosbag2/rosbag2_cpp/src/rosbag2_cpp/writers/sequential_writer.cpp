// Copyright 2018, Bosch Software Innovations GmbH.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rosbag2_cpp/writers/sequential_writer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rcpputils/filesystem_helper.hpp"

#include "rosbag2_cpp/info.hpp"
#include "rosbag2_cpp/logging.hpp"

#include "rosbag2_storage/storage_options.hpp"

namespace rosbag2_cpp {
namespace writers {

static constexpr char const* kDefaultStorageID = "mcap";

namespace {
std::string strip_parent_path(const std::string& relative_path) {
    return rcpputils::fs::path(relative_path).filename().string();
}
}  // namespace

SequentialWriter::SequentialWriter(std::unique_ptr<rosbag2_storage::StorageFactoryInterface> storage_factory, std::shared_ptr<SerializationFormatConverterFactoryInterface> converter_factory,
                                   std::unique_ptr<rosbag2_storage::MetadataIo> metadata_io)
    : storage_factory_(std::move(storage_factory)),
      converter_factory_(std::move(converter_factory)),
      storage_(nullptr),
      metadata_io_(std::move(metadata_io)),
      converter_(nullptr),
      topics_names_to_info_(),
      metadata_() {}

SequentialWriter::~SequentialWriter() {
    close();
}

void SequentialWriter::init_metadata() {
    metadata_ = rosbag2_storage::BagMetadata{};
    metadata_.storage_identifier = storage_->get_storage_identifier();
    metadata_.starting_time = std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::nanoseconds::max());
    metadata_.relative_file_paths = {strip_parent_path(storage_->get_relative_file_path())};
    rosbag2_storage::FileInformation file_info{};
    file_info.path = strip_parent_path(storage_->get_relative_file_path());
    file_info.starting_time = std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::nanoseconds::max());
    file_info.message_count = 0;
    metadata_.files = {file_info};
}

void SequentialWriter::open(const rosbag2_storage::StorageOptions& storage_options, const ConverterOptions& converter_options) {
    storage_options_ = storage_options;
    auto now = std::chrono::system_clock::now();
    if ("" == storage_options.uri) {
        // std::time_t time_now = std::chrono::system_clock::to_time_t(now);
        // std::tm tm_now = *std::localtime(&time_now);
        // std::ostringstream oss;
        // oss << std::put_time(&tm_now, "%Y-%m-%d_%H-%M-%S");
        // std::string time_str = oss.str();
        // storage_options_.uri = "./" + time_str;
        storage_name_ = "";
        base_folder_ = "./";
    } else if (storage_options.uri.back() == '/') {
        storage_name_ = "";
        base_folder_ = storage_options.uri;
    } else {
        storage_name_ = rcpputils::fs::path(storage_options_.uri).filename().string();
        base_folder_ = rcpputils::fs::path(storage_options_.uri).parent_path().string() + "/";
    }

    if (storage_options_.storage_id.empty()) {
        storage_options_.storage_id = kDefaultStorageID;
    }

    if (converter_options.output_serialization_format != converter_options.input_serialization_format) {
        converter_ = std::make_unique<Converter>(converter_options, converter_factory_);
    }

    rcpputils::fs::path db_path = rcpputils::fs::path(base_folder_);
    if (!rcpputils::fs::exists(db_path)) {
        bool dir_created = rcpputils::fs::create_directories(db_path);
        if (!dir_created) {
            std::stringstream error;
            error << "Failed to create database directory (" << db_path.string() << ").";
            throw std::runtime_error{error.str()};
        }
    }

    storage_options_.uri = format_storage_uri(base_folder_, storage_name_, 0);
    storage_ = storage_factory_->open_read_write(storage_options_, std::bind(&SequentialWriter::handle_record_event, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    if (!storage_) {
        throw std::runtime_error("No storage could be initialized. Abort");
    }

    if (storage_options_.max_bagfile_size != 0 && storage_options_.max_bagfile_size < storage_->get_minimum_split_file_size()) {
        std::stringstream error;
        error << "Invalid bag splitting size given. Please provide a value greater than " << storage_->get_minimum_split_file_size() << ". Specified value of " << storage_options.max_bagfile_size;
        throw std::runtime_error{error.str()};
    }
    use_cache_ = storage_options.max_cache_size > 0u;
    if (storage_options.snapshot_mode && !use_cache_) {
        throw std::runtime_error("Max cache size must be greater than 0 when snapshot mode is enabled");
    }

    if (use_cache_) {
        if (storage_options.snapshot_mode) {
            message_cache_ = std::make_shared<rosbag2_cpp::cache::CircularMessageCache>(storage_options.max_cache_size);
        } else {
            message_cache_ = std::make_shared<rosbag2_cpp::cache::MessageCache>(storage_options.max_cache_size);
        }
        cache_consumer_ = std::make_unique<rosbag2_cpp::cache::CacheConsumer>(message_cache_, std::bind(&SequentialWriter::write_messages, this, std::placeholders::_1));
    }
    init_metadata();
    if ("mcap" == storage_options_.storage_id) {
        //上报消息
        rcpputils::fs::path abs_path(storage_->get_relative_file_path());
        auto info = std::make_shared<bag_events::BagSplitInfo>();
        info->opened_file = abs_path.absolute_path();
        // 将当前时间点转换为纳秒级时间戳
        auto nano_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        info->na_time = nano_seconds;
        callback_manager_.execute_callbacks(bag_events::BagEvent::WRITE_SPLIT, info);
        // 写入附件队列全部附件
        for (auto& attachment : attachments_list_) {
            add_attach(attachment);
        }
    }

    // storage_->RegisterSplitFileCallback(std::bind(&SequentialWriter::handle_record_event, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

void SequentialWriter::close() {
    if (use_cache_) {
        // destructor will flush message cache
        cache_consumer_.reset();
        message_cache_.reset();
    }

    // if ("mcap" == storage_options_.storage_id) {
    //     if (!base_folder_.empty()) {
    //         finalize_metadata();
    //         metadata_io_->write_metadata(base_folder_, metadata_);
    //     }
    // }
    auto info = std::make_shared<bag_events::BagSplitInfo>();
    if (storage_ != nullptr) {
        rcpputils::fs::path abs_path(storage_->get_relative_file_path());
        info->closed_file = abs_path.absolute_path();
    }
    auto now = std::chrono::system_clock::now();
    auto nano_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    info->na_time = nano_seconds;
    storage_.reset();  // Necessary to ensure that the storage is destroyed before the factory
    storage_factory_.reset();
    //去掉包名的.active后缀
    pre_bag_remove_active_suffix(info->closed_file);
    //上报文件关闭
    callback_manager_.execute_callbacks(bag_events::BagEvent::WRITE_SPLIT, info);
}

void SequentialWriter::create_topic(const rosbag2_storage::TopicMetadata& topic_with_type) {
    if (topics_names_to_info_.find(topic_with_type.name) != topics_names_to_info_.end()) {
        // nothing to do, topic already created
        return;
    }

    if (!storage_) {
        throw std::runtime_error("Bag is not open. Call open() before writing.");
    }

    rosbag2_storage::TopicInformation info{};
    info.topic_metadata = topic_with_type;

    bool insert_succeded = false;
    {
        std::lock_guard<std::mutex> lock(topics_info_mutex_);
        const auto insert_res = topics_names_to_info_.insert(std::make_pair(topic_with_type.name, info));
        insert_succeded = insert_res.second;
    }

    if (!insert_succeded) {
        std::stringstream errmsg;
        errmsg << "Failed to insert topic \"" << topic_with_type.name << "\"!";

        throw std::runtime_error(errmsg.str());
    }
    storage_->create_topic(topic_with_type);
    if (converter_) {
        converter_->add_topic(topic_with_type.name, topic_with_type.type);
    }
}

void SequentialWriter::remove_topic(const rosbag2_storage::TopicMetadata& topic_with_type) {
    if (!storage_) {
        throw std::runtime_error("Bag is not open. Call open() before removing.");
    }

    bool erased = false;
    {
        std::lock_guard<std::mutex> lock(topics_info_mutex_);
        erased = topics_names_to_info_.erase(topic_with_type.name) > 0;
    }

    if (erased) {
        storage_->remove_topic(topic_with_type);
    } else {
        std::stringstream errmsg;
        errmsg << "Failed to remove the non-existing topic \"" << topic_with_type.name << "\"!";

        throw std::runtime_error(errmsg.str());
    }
}

std::string SequentialWriter::format_storage_uri(const std::string& base_folder, const std::string& storage_name, uint64_t storage_count, const std::string& time_suffix) {
    std::string time_str;
    if ("" == time_suffix) {
        auto now = std::chrono::system_clock::now();
        std::time_t time_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now = *std::localtime(&time_now);
        std::ostringstream oss;
        oss << std::put_time(&tm_now, "%Y%m%d-%H%M%S");
        time_str = oss.str();
    } else {
        time_str = time_suffix;
    }

    // storage_options_.uri = "./" + time_str;

    std::stringstream storage_file_name;
    if ("" == storage_name) {
        storage_file_name << base_folder << time_str << "_" << storage_count;
    } else if (storage_options_.use_time_suffix) {
        storage_file_name << base_folder << storage_name << "_" << time_str << "_" << storage_count;
    } else {
        storage_file_name << base_folder << storage_name << "_" << storage_count;
    }
    return storage_file_name.str();
}

void SequentialWriter::switch_to_next_storage(const std::string& time_suffix) {
    // consume remaining message cache
    if (use_cache_) {
        cache_consumer_->stop();
        message_cache_->log_dropped();
    }

    storage_options_.uri = format_storage_uri(base_folder_, storage_name_, metadata_.file_index, time_suffix);
    storage_ = storage_factory_->open_read_write(storage_options_);

    if (!storage_) {
        std::stringstream errmsg;
        errmsg << "Failed to rollover bagfile to new file: \"" << storage_options_.uri << "\"!";

        throw std::runtime_error(errmsg.str());
    }

    // 切包时写入附件队列全部附件到新的包
    for (auto& attachment : attachments_list_) {
        add_attach(attachment);
    }

    // Re-register all topics since we rolled-over to a new bagfile.
    for (const auto& topic : topics_names_to_info_) {
        storage_->create_topic(topic.second.topic_metadata);
    }

    if (use_cache_) {
        // restart consumer thread for cache
        cache_consumer_->start();
    }
}

void SequentialWriter::split_bagfile(const std::string& time_suffix) {
    auto info = std::make_shared<bag_events::BagSplitInfo>();
    rcpputils::fs::path per_path(storage_->get_relative_file_path());
    info->closed_file = per_path.absolute_path();
    switch_to_next_storage(time_suffix);
    pre_bag_remove_active_suffix(info->closed_file);
    metadata_.relative_file_paths.push_back(strip_parent_path(storage_->get_relative_file_path()));
    rosbag2_storage::FileInformation file_info{};
    file_info.path = strip_parent_path(storage_->get_relative_file_path());
    metadata_.files.push_back(file_info);
    metadata_.file_index = (metadata_.file_index + 1);

    //上报event
    rcpputils::fs::path abs_path(storage_->get_relative_file_path());
    info->opened_file = abs_path.absolute_path();
    auto now = std::chrono::system_clock::now();
    auto nano_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    info->na_time = nano_seconds;
    callback_manager_.execute_callbacks(bag_events::BagEvent::WRITE_SPLIT, info);
}

void SequentialWriter::pre_bag_remove_active_suffix(std::string& per_path) {
    size_t extensionPos = per_path.find(".active");
    if (extensionPos != std::string::npos) {
        std::string new_path = per_path;
        new_path = per_path.substr(0, extensionPos);
        int result = std::rename(per_path.c_str(), new_path.c_str());
        if (result != 0) {
            COMMON_LOG_ERROR << "rename " << per_path << " failed.";
        }
    }
}

void SequentialWriter::deleteOldestFile() {
    std::string oldestFileName = metadata_.relative_file_paths[0];
    const std::string suffix = ".active";

    std::string oldestFilePath = (rcpputils::fs::path(base_folder_) / oldestFileName).string();
    size_t suffixIndex = oldestFilePath.rfind(suffix);
    if (suffixIndex == oldestFilePath.size() - suffix.size()) {
        oldestFilePath = oldestFilePath.substr(0, suffixIndex);
    }
    if (!rcpputils::fs::exists(rcpputils::fs::path(oldestFilePath))) {
        std::stringstream errmsg;
        COMMON_LOG_ERROR << oldestFilePath << " not exists. Can't be delete.";
    }
    if (remove(oldestFilePath.c_str()) != 0) {
        COMMON_LOG_ERROR << "remove the oldest file:" << oldestFilePath << " failed.";
    } else {
        metadata_.files.erase(metadata_.files.begin());
        metadata_.relative_file_paths.erase(metadata_.relative_file_paths.begin());
    }
}

void SequentialWriter::write(std::shared_ptr<rosbag2_storage::SerializedBagMessage> message) {
    if (!storage_) {
        throw std::runtime_error("Bag is not open. Call open() before writing.");
    }

    // Get TopicInformation handler for counting messages.
    rosbag2_storage::TopicInformation* topic_information{nullptr};
    try {
        topic_information = &topics_names_to_info_.at(message->topic_name);
    } catch (const std::out_of_range& /* oor */) {
        std::stringstream errmsg;
        errmsg << "Failed to write on topic '" << message->topic_name << "'. Call create_topic() before first write.";
        throw std::runtime_error(errmsg.str());
    }

    const auto message_timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::nanoseconds(message->time_stamp));

    if ("mcap" == storage_options_.storage_id) {
        if (should_split_bagfile(message_timestamp)) {
            std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(message_timestamp.time_since_epoch()).count();
            std::tm tmTime = *std::localtime(&today_time);
            // 格式化为字符串
            std::ostringstream oss;
            oss << std::put_time(&tmTime, "%Y%m%d-%H%M%S");
            std::string time_suffix = oss.str();
            split_bagfile(time_suffix);
            // Update bagfile starting time
            metadata_.starting_time = message_timestamp;
            metadata_.files.back().starting_time = message_timestamp;
            uint16_t max_store_files = storage_options_.max_files;
            if ((metadata_.files.size() > max_store_files) || (metadata_.relative_file_paths.size() > max_store_files)) {
                deleteOldestFile();
            }
            if (storage_options_.split_bag_now) {
                storage_options_.split_bag_now = false;
            }
        }
    }

    metadata_.starting_time = std::min(metadata_.starting_time, message_timestamp);

    metadata_.files.back().starting_time = std::min(metadata_.files.back().starting_time, message_timestamp);
    const auto duration = message_timestamp - metadata_.starting_time;
    metadata_.duration = std::max(metadata_.duration, duration);

    const auto file_duration = message_timestamp - metadata_.files.back().starting_time;
    metadata_.files.back().duration = std::max(metadata_.files.back().duration, file_duration);

    auto converted_msg = get_writeable_message(message);

    metadata_.files.back().message_count++;

    if (storage_options_.max_cache_size == 0u) {
        // If cache size is set to zero, we write to storage directly
        storage_->write(converted_msg);
        ++topic_information->message_count;
    } else {
        // Otherwise, use cache buffer
        message_cache_->push(converted_msg);
    }
}

bool SequentialWriter::take_snapshot() {
    if (!storage_options_.snapshot_mode) {
        ROSBAG2_CPP_LOG_WARN << "SequentialWriter take_snaphot called when snapshot mode is disabled";
        return false;
    }
    message_cache_->notify_data_ready();
    return true;
}

std::shared_ptr<rosbag2_storage::SerializedBagMessage> SequentialWriter::get_writeable_message(std::shared_ptr<rosbag2_storage::SerializedBagMessage> message) {
    // return converter_ ? converter_->convert(message) : message;
    return message;
}

bool SequentialWriter::should_split_bagfile(const std::chrono::time_point<std::chrono::high_resolution_clock>& current_time) const {
    // Assume we aren't splitting
    bool should_split = false;

    //Splitting by outside call
    if (storage_options_.split_bag_now) {
        return true;
    }

    // Splitting by size
    if (storage_options_.max_bagfile_size != rosbag2_storage::storage_interfaces::MAX_BAGFILE_SIZE_NO_SPLIT) {
        should_split = (storage_->get_bagfile_size() >= storage_options_.max_bagfile_size);
    }

    // Splitting by time
    if (storage_options_.max_bagfile_duration != rosbag2_storage::storage_interfaces::MAX_BAGFILE_DURATION_NO_SPLIT) {
        auto max_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(storage_options_.max_bagfile_duration));
        should_split = should_split || ((current_time - metadata_.starting_time) > max_duration_ns);
    }

    return should_split;
}

void SequentialWriter::finalize_metadata() {
    metadata_.bag_size = 0;

    for (const auto& path : metadata_.relative_file_paths) {
        const auto bag_path = rcpputils::fs::path{path};

        if (bag_path.exists()) {
            metadata_.bag_size += bag_path.file_size();
        }
    }

    metadata_.topics_with_message_count.clear();
    metadata_.topics_with_message_count.reserve(topics_names_to_info_.size());
    metadata_.message_count = 0;

    for (const auto& topic : topics_names_to_info_) {
        metadata_.topics_with_message_count.push_back(topic.second);
        metadata_.message_count += topic.second.message_count;
    }
}

void SequentialWriter::write_messages(const std::vector<std::shared_ptr<const rosbag2_storage::SerializedBagMessage>>& messages) {
    if (messages.empty()) {
        return;
    }
    storage_->write(messages);
    std::lock_guard<std::mutex> lock(topics_info_mutex_);
    for (const auto& msg : messages) {
        if (topics_names_to_info_.find(msg->topic_name) != topics_names_to_info_.end()) {
            topics_names_to_info_[msg->topic_name].message_count++;
        }
    }
}

void SequentialWriter::add_event_callbacks(const bag_events::WriterEventCallbacks& callbacks) {
    if (callbacks.write_split_callback) {
        callback_manager_.add_event_callback(callbacks.write_split_callback, bag_events::BagEvent::WRITE_SPLIT);
    }
}

void SequentialWriter::handle_record_event(std::string absolut_filr_path, uint64_t time, bool is_open) {
    auto info = std::make_shared<bag_events::BagSplitInfo>();
    if (is_open) {
        info->opened_file = absolut_filr_path;
    } else {
        info->closed_file = absolut_filr_path;
    }
    info->na_time = time;
    callback_manager_.execute_callbacks(bag_events::BagEvent::WRITE_SPLIT, info);
}

void SequentialWriter::add_attach(std::shared_ptr<rosbag2_storage::Attachment> attachment) {
    if (!storage_) {
        throw std::runtime_error("Bag is not open. Call open() before add attachments.");
    }
    storage_->add_attach(attachment);
}

void SequentialWriter::add_attach_to_list(std::shared_ptr<rosbag2_storage::Attachment>& attachment) {
    attachments_list_.emplace_back(attachment);
}

void SequentialWriter::splite_bag_now() {
    storage_options_.split_bag_now = true;
}
}  // namespace writers
}  // namespace rosbag2_cpp
