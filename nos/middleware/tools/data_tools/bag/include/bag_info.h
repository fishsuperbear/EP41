#pragma once
#include <chrono>
#include <iostream>
#include <vector>
#include "rosbag2_storage/bag_metadata.hpp"

namespace hozon {
namespace netaos {
namespace bag {

struct InfoOptions {
   public:
    bool show_help_info = false;
    std::vector<std::string> bag_files;
};

void ShowBagInfo(const InfoOptions& infoOptions);
rosbag2_storage::BagMetadata GetBagInfo(const std::string& bag_file, std::string storage_id = "mcap");

}  // namespace bag
}  //namespace netaos
}  //namespace hozon