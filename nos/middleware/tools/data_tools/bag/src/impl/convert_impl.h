#pragma once
#include <iostream>
#include <memory>
#include <rosbag2_cpp/writer.hpp>
#include "convert.h"

namespace hozon {
namespace netaos {
namespace bag {

class ConvertImpl {
   private:
    std::unique_ptr<rosbag2_cpp::Writer> _writer;
    uint _count;
    std::map<std::string, std::string> topic_type_info_;

   public:
    void Start(ConvertOptions convert_option);

    void Stop();
    void WriteMessage(std::string topic_name, std::string data_type, int64_t time, std::vector<std::uint8_t> data);
    void WriteBagMessage(std::shared_ptr<rosbag2_storage::SerializedBagMessage> data);
    ConvertImpl();
    ~ConvertImpl();

    static bool is_stop;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon