#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace topic {
class ListImpl;

struct ListOptions {
    bool method = false;
};

class List {
   public:
    List();
    void Start(ListOptions list_options);
    ~List();

   private:
    std::unique_ptr<ListImpl> list_impl_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon