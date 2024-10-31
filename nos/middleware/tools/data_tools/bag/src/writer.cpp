#include "writer.h"
#include <impl/writer_impl.h>

namespace hozon {
namespace netaos {
namespace bag {

Writer::Writer() {
    writer_impl_ = std::make_unique<WriterImpl>();
};

Writer::~Writer(){};

WriterErrorCode Writer::Open(const WriterOptions& record_options) {
    return writer_impl_->Open(record_options);
};

void Writer::write(const std::string& topic_name, const std::string& proto_name, const std::string& serialized_string, const int64_t& time, const std::string& idl_tpye) {
    writer_impl_->write(topic_name, proto_name, serialized_string, time, idl_tpye);
}

void Writer::WriterRegisterCallback(const WriterCallback& callback) {
    writer_impl_->WriterRegisterCallback(callback);
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
