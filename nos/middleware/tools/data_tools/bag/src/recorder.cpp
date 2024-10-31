#include <recorder.h>
#include <impl/recorder_impl.h>

namespace hozon {
namespace netaos {
namespace bag {

Recorder::Recorder() {
    recorder_impl_ = std::make_unique<RecorderImpl>();
}

Recorder::~Recorder() {
    if (recorder_impl_) {
        recorder_impl_ = nullptr;
    }
}

RecordErrorCode Recorder::Start(const RecordOptions& recordOptions) {
    return recorder_impl_->Start(recordOptions);
}

RecordErrorCode Recorder::Stop() {
    recorder_impl_->Stop();
    return RecordErrorCode::SUCCESS;
}

void Recorder::RecorderRegisterCallback(const RecorderCallback& callback) {
    recorder_impl_->RecorderRegisterCallback(callback);
}

void Recorder::SpliteBagNow() {
    recorder_impl_->SpliteBagNow();
}

void Recorder::RegisterPreWriteCallbak(const std::string& topic_name, const PreWriteCallbak& data_handler) {
    recorder_impl_->RegisterPreWriteCallbak(topic_name, data_handler);
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon