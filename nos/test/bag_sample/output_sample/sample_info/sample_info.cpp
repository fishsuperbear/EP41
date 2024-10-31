#include <cstring>
#include "bag_info.h"

int main(int argc, char** argv) {
    //获取.mcap文件信息
    hozon::netaos::bag::BagMetadata data = hozon::netaos::bag::GetBagInfo("/home/sw/work/netaos/2023-07-31_10-20-51/2023-07-31_10-20-51_0.mcap");
    std::cout << "Bag size:          " << data.bag_size << " Byte" << std::endl;
    std::cout << "version:           " << data.version << std::endl;
    std::cout << "Storage id:        " << data.storage_identifier << std::endl;
    std::cout << "Duration:          " << std::chrono::duration_cast<std::chrono::duration<double>>(data.duration).count() << " s" << std::endl;
    std::cout << "compression:       " << data.compression_format << std::endl;
    std::cout << "Messages:          " << data.message_count << std::endl;
    for (size_t i = 0; i < data.topics_with_message_count.size(); i++) {
        if (0 == i) {
            std::cout << "Event information: Event: " << data.topics_with_message_count[i].topic_metadata.name << " | Type: " << data.topics_with_message_count[i].topic_metadata.type
                      << " | Count: " << data.topics_with_message_count[i].message_count << " | Serialization Format: " << data.topics_with_message_count[i].topic_metadata.serialization_format
                      << std::endl;
        } else {
            std::cout << "                   Event: " << data.topics_with_message_count[i].topic_metadata.name << " | Type: " << data.topics_with_message_count[i].topic_metadata.type
                      << " | Count: " << data.topics_with_message_count[i].message_count << " | Serialization Format: " << data.topics_with_message_count[i].topic_metadata.serialization_format
                      << std::endl;
        }
    }

    for (size_t i = 0; i < data.files.size(); i++) {
        if (0 == i) {
            std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(data.files[i].starting_time.time_since_epoch()).count();
            char* dt = ctime(&today_time);
            dt[std::strlen(dt) - 1] = 0;
            std::cout << "Files:              Path: " << data.files[i].path << " | Starting time: " << dt
                      << " | Duration: " << std::chrono::duration_cast<std::chrono::duration<double>>(data.files[i].duration).count() << " | message_count: " << data.files[i].message_count
                      << std::endl;
        } else {
            std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(data.files[i].starting_time.time_since_epoch()).count();
            char* dt = ctime(&today_time);
            dt[std::strlen(dt) - 1] = 0;
            std::cout << "                    Path: " << data.files[i].path << " | Starting time: " << dt
                      << " | Duration: " << std::chrono::duration_cast<std::chrono::duration<double>>(data.files[i].duration).count() << " | message_count: " << data.files[i].message_count
                      << std::endl;
        }
    }

    std::cout << "--------------------------------------------------" << std::endl;
    //获取mcap文件夹信息
    hozon::netaos::bag::BagMetadata data2 = hozon::netaos::bag::GetBagInfo("/home/sw/work/netaos/2023-07-31_10-20-51/");
    std::cout << "Bag size:          " << data2.bag_size << " Byte" << std::endl;
    std::cout << "version:           " << data2.version << std::endl;
    std::cout << "Storage id:        " << data2.storage_identifier << std::endl;
    std::cout << "Duration:          " << std::chrono::duration_cast<std::chrono::duration<double>>(data2.duration).count() << " s" << std::endl;
    std::cout << "compression:       " << data2.compression_format << std::endl;
    std::cout << "Messages:          " << data2.message_count << std::endl;
    for (size_t i = 0; i < data2.topics_with_message_count.size(); i++) {
        if (0 == i) {
            std::cout << "Event information: Event: " << data2.topics_with_message_count[i].topic_metadata.name << " | Type: " << data2.topics_with_message_count[i].topic_metadata.type
                      << " | Count: " << data2.topics_with_message_count[i].message_count << " | Serialization Format: " << data2.topics_with_message_count[i].topic_metadata.serialization_format
                      << std::endl;
        } else {
            std::cout << "                   Event: " << data2.topics_with_message_count[i].topic_metadata.name << " | Type: " << data2.topics_with_message_count[i].topic_metadata.type
                      << " | Count: " << data2.topics_with_message_count[i].message_count << " | Serialization Format: " << data2.topics_with_message_count[i].topic_metadata.serialization_format
                      << std::endl;
        }
    }

    for (size_t i = 0; i < data2.files.size(); i++) {
        if (0 == i) {
            std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(data2.files[i].starting_time.time_since_epoch()).count();
            char* dt = ctime(&today_time);
            dt[std::strlen(dt) - 1] = 0;
            std::cout << "Files:              Path: " << data2.files[i].path << " | Starting time: " << dt
                      << " | Duration: " << std::chrono::duration_cast<std::chrono::duration<double>>(data2.files[i].duration).count() << " | message_count: " << data2.files[i].message_count
                      << std::endl;
        } else {
            std::time_t today_time = std::chrono::duration_cast<std::chrono::seconds>(data2.files[i].starting_time.time_since_epoch()).count();
            char* dt = ctime(&today_time);
            dt[std::strlen(dt) - 1] = 0;
            std::cout << "                    Path: " << data2.files[i].path << " | Starting time: " << dt
                      << " | Duration: " << std::chrono::duration_cast<std::chrono::duration<double>>(data2.files[i].duration).count() << " | message_count: " << data2.files[i].message_count
                      << std::endl;
        }
    }

    return 0;
}