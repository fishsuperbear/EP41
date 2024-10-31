#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace bag {

class SaveImpl;

enum SaveErrorCode {
    SAVE_SUCCESS = 0,
    FAILED = 1,                    //失败
    FAILED_NO_TOPIC_SPECIFIED = 2  //没有指定要转化的topic
};

struct SaveOptions {
    std::string url = "";
    std::vector<std::string> topics;
};

class Save {
   public:
    SaveErrorCode Start(SaveOptions convert_option);
    Save();
    ~Save();

   private:
    std::unique_ptr<SaveImpl> save_impl_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon