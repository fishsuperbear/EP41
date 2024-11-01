// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: modules/common_msgs/demo_msgs/demo.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3014000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3014000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "modules/common_msgs/basic_msgs/header.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto;
namespace netaos {
namespace common {
namespace demo {
class Demo;
class DemoDefaultTypeInternal;
extern DemoDefaultTypeInternal _Demo_default_instance_;
}  // namespace demo
}  // namespace common
}  // namespace netaos
PROTOBUF_NAMESPACE_OPEN
template<> ::netaos::common::demo::Demo* Arena::CreateMaybeMessage<::netaos::common::demo::Demo>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace netaos {
namespace common {
namespace demo {

// ===================================================================

class Demo PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:netaos.common.demo.Demo) */ {
 public:
  inline Demo() : Demo(nullptr) {}
  virtual ~Demo();

  Demo(const Demo& from);
  Demo(Demo&& from) noexcept
    : Demo() {
    *this = ::std::move(from);
  }

  inline Demo& operator=(const Demo& from) {
    CopyFrom(from);
    return *this;
  }
  inline Demo& operator=(Demo&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const Demo& default_instance();

  static inline const Demo* internal_default_instance() {
    return reinterpret_cast<const Demo*>(
               &_Demo_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Demo& a, Demo& b) {
    a.Swap(&b);
  }
  inline void Swap(Demo* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Demo* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Demo* New() const final {
    return CreateMaybeMessage<Demo>(nullptr);
  }

  Demo* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Demo>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Demo& from);
  void MergeFrom(const Demo& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Demo* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "netaos.common.demo.Demo";
  }
  protected:
  explicit Demo(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto);
    return ::descriptor_table_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kMsgFieldNumber = 2,
    kHeaderFieldNumber = 1,
  };
  // optional string msg = 2;
  bool has_msg() const;
  private:
  bool _internal_has_msg() const;
  public:
  void clear_msg();
  const std::string& msg() const;
  void set_msg(const std::string& value);
  void set_msg(std::string&& value);
  void set_msg(const char* value);
  void set_msg(const char* value, size_t size);
  std::string* mutable_msg();
  std::string* release_msg();
  void set_allocated_msg(std::string* msg);
  private:
  const std::string& _internal_msg() const;
  void _internal_set_msg(const std::string& value);
  std::string* _internal_mutable_msg();
  public:

  // optional .netaos.common.Header header = 1;
  bool has_header() const;
  private:
  bool _internal_has_header() const;
  public:
  void clear_header();
  const ::netaos::common::Header& header() const;
  ::netaos::common::Header* release_header();
  ::netaos::common::Header* mutable_header();
  void set_allocated_header(::netaos::common::Header* header);
  private:
  const ::netaos::common::Header& _internal_header() const;
  ::netaos::common::Header* _internal_mutable_header();
  public:
  void unsafe_arena_set_allocated_header(
      ::netaos::common::Header* header);
  ::netaos::common::Header* unsafe_arena_release_header();

  // @@protoc_insertion_point(class_scope:netaos.common.demo.Demo)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr msg_;
  ::netaos::common::Header* header_;
  friend struct ::TableStruct_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Demo

// optional .netaos.common.Header header = 1;
inline bool Demo::_internal_has_header() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  PROTOBUF_ASSUME(!value || header_ != nullptr);
  return value;
}
inline bool Demo::has_header() const {
  return _internal_has_header();
}
inline const ::netaos::common::Header& Demo::_internal_header() const {
  const ::netaos::common::Header* p = header_;
  return p != nullptr ? *p : reinterpret_cast<const ::netaos::common::Header&>(
      ::netaos::common::_Header_default_instance_);
}
inline const ::netaos::common::Header& Demo::header() const {
  // @@protoc_insertion_point(field_get:netaos.common.demo.Demo.header)
  return _internal_header();
}
inline void Demo::unsafe_arena_set_allocated_header(
    ::netaos::common::Header* header) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(header_);
  }
  header_ = header;
  if (header) {
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:netaos.common.demo.Demo.header)
}
inline ::netaos::common::Header* Demo::release_header() {
  _has_bits_[0] &= ~0x00000002u;
  ::netaos::common::Header* temp = header_;
  header_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::netaos::common::Header* Demo::unsafe_arena_release_header() {
  // @@protoc_insertion_point(field_release:netaos.common.demo.Demo.header)
  _has_bits_[0] &= ~0x00000002u;
  ::netaos::common::Header* temp = header_;
  header_ = nullptr;
  return temp;
}
inline ::netaos::common::Header* Demo::_internal_mutable_header() {
  _has_bits_[0] |= 0x00000002u;
  if (header_ == nullptr) {
    auto* p = CreateMaybeMessage<::netaos::common::Header>(GetArena());
    header_ = p;
  }
  return header_;
}
inline ::netaos::common::Header* Demo::mutable_header() {
  // @@protoc_insertion_point(field_mutable:netaos.common.demo.Demo.header)
  return _internal_mutable_header();
}
inline void Demo::set_allocated_header(::netaos::common::Header* header) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(header_);
  }
  if (header) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(header)->GetArena();
    if (message_arena != submessage_arena) {
      header = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, header, submessage_arena);
    }
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  header_ = header;
  // @@protoc_insertion_point(field_set_allocated:netaos.common.demo.Demo.header)
}

// optional string msg = 2;
inline bool Demo::_internal_has_msg() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Demo::has_msg() const {
  return _internal_has_msg();
}
inline void Demo::clear_msg() {
  msg_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& Demo::msg() const {
  // @@protoc_insertion_point(field_get:netaos.common.demo.Demo.msg)
  return _internal_msg();
}
inline void Demo::set_msg(const std::string& value) {
  _internal_set_msg(value);
  // @@protoc_insertion_point(field_set:netaos.common.demo.Demo.msg)
}
inline std::string* Demo::mutable_msg() {
  // @@protoc_insertion_point(field_mutable:netaos.common.demo.Demo.msg)
  return _internal_mutable_msg();
}
inline const std::string& Demo::_internal_msg() const {
  return msg_.Get();
}
inline void Demo::_internal_set_msg(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  msg_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline void Demo::set_msg(std::string&& value) {
  _has_bits_[0] |= 0x00000001u;
  msg_.Set(
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:netaos.common.demo.Demo.msg)
}
inline void Demo::set_msg(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000001u;
  msg_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(value), GetArena());
  // @@protoc_insertion_point(field_set_char:netaos.common.demo.Demo.msg)
}
inline void Demo::set_msg(const char* value,
    size_t size) {
  _has_bits_[0] |= 0x00000001u;
  msg_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:netaos.common.demo.Demo.msg)
}
inline std::string* Demo::_internal_mutable_msg() {
  _has_bits_[0] |= 0x00000001u;
  return msg_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* Demo::release_msg() {
  // @@protoc_insertion_point(field_release:netaos.common.demo.Demo.msg)
  if (!_internal_has_msg()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return msg_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void Demo::set_allocated_msg(std::string* msg) {
  if (msg != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  msg_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), msg,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:netaos.common.demo.Demo.msg)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace demo
}  // namespace common
}  // namespace netaos

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_modules_2fcommon_5fmsgs_2fdemo_5fmsgs_2fdemo_2eproto
