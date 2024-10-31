#ifndef CYBER_RECORD_FILE_SECTION_H_
#define CYBER_RECORD_FILE_SECTION_H_

namespace netaos {
namespace framework {
namespace record {

struct Section {
  proto::SectionType type;
  int64_t size;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_RECORD_FILE_SECTION_H_
