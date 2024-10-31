
#pragma once

#include <dirent.h>
#include <fcntl.h>
#include <glob.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

#define SAME_ELEMENT_OF_IPTR(PTR_ONE, PTR_TWO) \
  ((PTR_ONE).get() == (PTR_TWO).get())

#define NULL_IPTR(PTR) (nullptr == (PTR).get())

namespace hozon {
namespace fsmcore {

uint64_t now_usec();

bool get_proto_from_file(const std::string &file_name,
                         google::protobuf::Message *message);

bool get_proto_from_binary_file(const std::string &file_path,
                                google::protobuf::Message *message);

bool get_proto_from_ascii_file(const std::string &file_path,
                               google::protobuf::Message *message);

}  // namespace fsmcore
}  // namespace hozon