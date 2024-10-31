#pragma once
#include "save.h"

namespace hozon {
namespace netaos {
namespace bag {

class SaveImpl {
   private:
    int64_t _count = 0;

   public:
    SaveErrorCode Start(SaveOptions save_option);

    SaveImpl();
    ~SaveImpl();
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon