#include "convert.h"
#include "impl/convert_impl.h"

namespace hozon {
namespace netaos {
namespace bag {

Convert::Convert() {
    convert_impl_ = std::make_unique<ConvertImpl>();
}

Convert::~Convert() {}

void Convert::Stop() {
    convert_impl_->Stop();
};

void Convert::Start(ConvertOptions convert_option) {
    convert_impl_->Start(convert_option);
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon