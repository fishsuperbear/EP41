#include <bag_info.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <rosbag2_cpp/info.hpp>
#include "reader.h"

namespace hozon {
namespace netaos {
namespace bag {

rosbag2_storage::BagMetadata GetBagInfo(const std::string& bag_file, std::string storage_id) {
    // rosbag2_cpp::Info baginfo = rosbag2_cpp::Info();

    std::shared_ptr<Reader> reader = std::make_shared<Reader>();
    reader->Open(bag_file, storage_id);
    auto info = reader->GetMetadata();
    reader->Close();
    return info;
}

void ShowBagInfo(const InfoOptions& infoOptions) {
    for (auto bag_file : infoOptions.bag_files) {
        std::shared_ptr<Reader> reader = std::make_shared<Reader>();
        hozon::netaos::bag::ReaderErrorCode ret = reader->Open(bag_file, "mcap");
        if (hozon::netaos::bag::ReaderErrorCode::SUCCESS != ret) {
            return;
        }

        rosbag2_storage::BagMetadata data = reader->GetMetadata();
        reader->Close();

        std::cout << "Bag size:          " << data.bag_size << " Byte (" << (double)data.bag_size / 1024 / 1024 << "M)" << std::endl;
        std::cout << "version:           " << data.version << std::endl;
        std::cout << "app version:       " << data.app_version << std::endl;
        std::cout << "Storage id:        " << data.storage_identifier << std::endl;
        std::cout << "Duration:          " << std::chrono::duration_cast<std::chrono::duration<double>>(data.duration).count() << " s" << std::endl;

        long long start_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(data.starting_time.time_since_epoch()).count();
        std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(data.starting_time.time_since_epoch()).count();
        std::tm tmTime = *std::localtime(&today_time);
        // 格式化为字符串
        std::ostringstream oss;
        oss << std::put_time(&tmTime, "%Y-%m-%d-%H:%M:%S");
        std::cout << "Start time: " << oss.str() << " (" << start_nanoseconds << ")" << std::endl;
        long long end_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>((data.starting_time + data.duration).time_since_epoch()).count();
        today_time = std::chrono::duration_cast<std::chrono::seconds>((data.starting_time + data.duration).time_since_epoch()).count();
        tmTime = *std::localtime(&today_time);
        // 格式化为字符串
        std::ostringstream end_oss;
        end_oss << std::put_time(&tmTime, "%Y-%m-%d-%H:%M:%S");
        std::cout << "end   time: " << end_oss.str() << " (" << end_nanoseconds << ")" << std::endl;
        std::cout << "compression:       " << data.compression_format << std::endl;
        std::cout << "Messages:          " << data.message_count << std::endl;
        for (size_t i = 0; i < data.topics_with_message_count.size(); i++) {
            if (0 == i) {
                std::cout << "Topic information: Topic: " << data.topics_with_message_count[i].topic_metadata.name << " | Type: " << data.topics_with_message_count[i].topic_metadata.type
                          << " | Count: " << data.topics_with_message_count[i].message_count << " | Serialization Format: " << data.topics_with_message_count[i].topic_metadata.serialization_format
                          << std::endl;
            } else {
                std::cout << "                   Topic: " << data.topics_with_message_count[i].topic_metadata.name << " | Type: " << data.topics_with_message_count[i].topic_metadata.type
                          << " | Count: " << data.topics_with_message_count[i].message_count << " | Serialization Format: " << data.topics_with_message_count[i].topic_metadata.serialization_format
                          << std::endl;
            }
        }

        std::vector<rosbag2_storage::FileInformation> normal_file, cfg_file;
        std::map<std::string, std::vector<rosbag2_storage::FileInformation>> other_file_map;
        for (const auto& file_info : data.files) {
            if (file_info.file_type == "normal") {
                normal_file.emplace_back(file_info);
            } else if (file_info.file_type == "cfg") {
                cfg_file.emplace_back(file_info);
            } else {
                other_file_map[file_info.file_type].emplace_back(file_info);
            }
        }
        
        for (size_t i = 0; i < normal_file.size(); ++i) {
            std::cout << ((i == 0) ? "Attachments:        " : "                    ");
            std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(normal_file[i].starting_time.time_since_epoch()).count();
            std::tm tmTime = *std::localtime(&today_time);
            // 格式化为字符串
            std::ostringstream oss;
            oss << std::put_time(&tmTime, "%Y-%m-%d-%H:%M:%S");
            std::cout << "Name: " << normal_file[i].path << " | Record time: " << oss.str() << " | Type: " << normal_file[i].file_type << std::endl;
        }

        for (size_t i = 0; i < cfg_file.size(); ++i) {
            std::cout << ((i == 0) ? "Cfg:                " : "                    ");
            std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(cfg_file[i].starting_time.time_since_epoch()).count();
            std::tm tmTime = *std::localtime(&today_time);
            // 格式化为字符串
            std::ostringstream oss;
            oss << std::put_time(&tmTime, "%Y-%m-%d-%H:%M:%S");
            std::cout << "Name: " << cfg_file[i].path << " | Record time: " << oss.str() << " | Type: " << cfg_file[i].file_type << std::endl;
        }

        for (const auto& type : other_file_map) {
            for (size_t i = 0; i < type.second.size(); ++i) {
                std::string spaces(20 - type.first.size() - 1, ' ');
                std::cout << ((i == 0) ? (type.first + ":" + spaces) : "                    ");
                std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(type.second[i].starting_time.time_since_epoch()).count();
                std::tm tmTime = *std::localtime(&today_time);
                // 格式化为字符串
                std::ostringstream oss;
                oss << std::put_time(&tmTime, "%Y-%m-%d-%H:%M:%S");
                std::cout << "Name: " << type.second[i].path << " | Record time: " << oss.str() << " | Type: " << type.second[i].file_type << std::endl;
            }
        }
    }
}
}  // namespace bag
}  // namespace netaos
}  // namespace hozon