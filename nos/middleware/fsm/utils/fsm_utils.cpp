#include "fsm_utils.h"

namespace hozon {
namespace fsmcore {

uint64_t now_usec() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool get_proto_from_file(const std::string &file_path,
                         google::protobuf::Message *message) {
  // Try the binary parser first if it's much likely a binary proto.
  static const std::string kBinExt = ".bin";
  if (std::equal(kBinExt.rbegin(), kBinExt.rend(), file_path.rbegin())) {
    return get_proto_from_binary_file(file_path, message) ||
           get_proto_from_ascii_file(file_path, message);
  }

  return get_proto_from_ascii_file(file_path, message) ||
         get_proto_from_binary_file(file_path, message);
}

bool get_proto_from_ascii_file(const std::string &file_name,
                               google::protobuf::Message *message) {
  using google::protobuf::TextFormat;
  using google::protobuf::io::FileInputStream;
  using google::protobuf::io::ZeroCopyInputStream;
  int file_descriptor = open(file_name.c_str(), O_RDONLY);
  if (file_descriptor < 0) {
    std::cout << "Open file failed, file_path " << file_name << std::endl;
    return false;
  }

  ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
  bool success;
  try {
    success = TextFormat::Parse(input, message);
  } catch (...) {
    std::cout << "Google TextFormat::Parse file: " << file_name << "failed."
              << std::endl;
  }

  if (success) {
    std::cout << "Parse from file to proto message success, file_path "
              << file_name << std::endl;
  } else {
    std::cout << "Parse from file to proto message failed, file_path "
              << file_name << std::endl;
  }

  delete input;
  close(file_descriptor);
  return success;
}

bool get_proto_from_binary_file(const std::string &file_name,
                                google::protobuf::Message *message) {
  std::fstream input(file_name, std::ios::in | std::ios::binary);
  if (!input.good()) {
    std::cout << "Failed to open file " << file_name << " in binary mode.";
    return false;
  }
  if (!message->ParseFromIstream(&input)) {
    std::cout << "Failed to parse file " << file_name << " as binary proto.";
    return false;
  }
  return true;
}

}  // namespace fsmcore
}  // namespace hozon