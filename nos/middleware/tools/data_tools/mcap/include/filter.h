#pragma once
#include <iostream>
#include <vector>
#include "rosbag2_cpp/bag_events.hpp"

namespace hozon {
namespace netaos {
namespace mcap {

class Filter {
public:
    Filter();
    ~Filter();
    bool regular_matching_topic(std::vector<std::string> regular_topic_vec, std::string topic);
    void filter_base_topic_list(std::string mcap_file_path,
                                std::vector<std::string> regular_white_topic_vec,
                                std::vector<std::string> regular_black_topic_vec,
                                std::string output_folder_path);
    void set_output_file_path(rosbag2_cpp::bag_events::BagSplitInfo& info);
    std::string get_output_file_path();

private:
    std::string m_output_file_path;
};

}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
