#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace bag {

class ConvertImpl;

struct ConvertOptions {
    std::string input_file = "";
    std::string intput_file_type = "rtfbag";
    std::string output_file = "";
    std::string output_file_type = "mcap";
    std::string intput_data_version = "0228-0324";
    std::string output_data_version = "";
    std::vector<std::string> topics;
    std::vector<std::string> exclude_topics;
    bool use_time_suffix = true;
};

class Convert {
   public:
    void Start(ConvertOptions convert_option);
    void Stop();
    Convert();
    ~Convert();

   private:
    std::unique_ptr<ConvertImpl> convert_impl_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon