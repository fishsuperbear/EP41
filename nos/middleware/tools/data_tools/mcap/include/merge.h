#pragma once
#include <iostream>
#include <set>
#include "rosbag2_cpp/bag_events.hpp"

namespace hozon {
namespace netaos {
namespace mcap {

class Merge {
public:
    Merge();
    ~Merge();
    void merge_mcap(std::vector<std::string> mcap_file_path_vec,
                    std::vector<std::string> attachment_str_name_vec,
                    std::vector<std::string> attachment_str_vec,
                    std::vector<std::string> attachment_file_path_vec,
                    std::string output_file_path);
    void merge_mcap(std::vector<std::string> mcap_file_path_vec,
                    std::vector<std::string> attachment_file_path_vec,
                    std::string output_file_path);
    void merge_mcap(std::vector<std::string> mcap_file_path_vec,
                    std::vector<std::string> attachment_str_name_vec,
                    std::vector<std::string> attachment_str_vec,
                    std::string output_file_path);
    void merge_mcap(std::vector<std::string> mcap_file_path_vec,
                    std::string output_file_path);
    void set_output_file_path_vec(rosbag2_cpp::bag_events::BagSplitInfo& info);
    std::set<std::string> get_output_file_path_vec();

private:
    std::set<std::string> m_output_file_path_vec;
};

}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
