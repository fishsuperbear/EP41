// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: log_conf.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_log_5fconf_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_log_5fconf_2eproto

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
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_log_5fconf_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_log_5fconf_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_log_5fconf_2eproto;
namespace netaos {
namespace framework {
namespace proto {
class LogConfig;
class LogConfigDefaultTypeInternal;
extern LogConfigDefaultTypeInternal _LogConfig_default_instance_;
class MlogConfig;
class MlogConfigDefaultTypeInternal;
extern MlogConfigDefaultTypeInternal _MlogConfig_default_instance_;
}  // namespace proto
}  // namespace framework
}  // namespace netaos
PROTOBUF_NAMESPACE_OPEN
template<> ::netaos::framework::proto::LogConfig* Arena::CreateMaybeMessage<::netaos::framework::proto::LogConfig>(Arena*);
template<> ::netaos::framework::proto::MlogConfig* Arena::CreateMaybeMessage<::netaos::framework::proto::MlogConfig>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace netaos {
namespace framework {
namespace proto {

// ===================================================================

class MlogConfig PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:netaos.framework.proto.MlogConfig) */ {
 public:
  inline MlogConfig() : MlogConfig(nullptr) {}
  virtual ~MlogConfig();

  MlogConfig(const MlogConfig& from);
  MlogConfig(MlogConfig&& from) noexcept
    : MlogConfig() {
    *this = ::std::move(from);
  }

  inline MlogConfig& operator=(const MlogConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline MlogConfig& operator=(MlogConfig&& from) noexcept {
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
  static const MlogConfig& default_instance();

  static inline const MlogConfig* internal_default_instance() {
    return reinterpret_cast<const MlogConfig*>(
               &_MlogConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(MlogConfig& a, MlogConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(MlogConfig* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(MlogConfig* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline MlogConfig* New() const final {
    return CreateMaybeMessage<MlogConfig>(nullptr);
  }

  MlogConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<MlogConfig>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const MlogConfig& from);
  void MergeFrom(const MlogConfig& from);
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
  void InternalSwap(MlogConfig* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "netaos.framework.proto.MlogConfig";
  }
  protected:
  explicit MlogConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_log_5fconf_2eproto);
    return ::descriptor_table_log_5fconf_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kNameFieldNumber = 1,
    kLogLevelFieldNumber = 2,
    kSwFieldNumber = 3,
  };
  // required string name = 1;
  bool has_name() const;
  private:
  bool _internal_has_name() const;
  public:
  void clear_name();
  const std::string& name() const;
  void set_name(const std::string& value);
  void set_name(std::string&& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  std::string* mutable_name();
  std::string* release_name();
  void set_allocated_name(std::string* name);
  private:
  const std::string& _internal_name() const;
  void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // required string log_level = 2 [default = "info"];
  bool has_log_level() const;
  private:
  bool _internal_has_log_level() const;
  public:
  void clear_log_level();
  const std::string& log_level() const;
  void set_log_level(const std::string& value);
  void set_log_level(std::string&& value);
  void set_log_level(const char* value);
  void set_log_level(const char* value, size_t size);
  std::string* mutable_log_level();
  std::string* release_log_level();
  void set_allocated_log_level(std::string* log_level);
  private:
  const std::string& _internal_log_level() const;
  void _internal_set_log_level(const std::string& value);
  std::string* _internal_mutable_log_level();
  public:

  // optional bool sw = 3 [default = true];
  bool has_sw() const;
  private:
  bool _internal_has_sw() const;
  public:
  void clear_sw();
  bool sw() const;
  void set_sw(bool value);
  private:
  bool _internal_sw() const;
  void _internal_set_sw(bool value);
  public:

  // @@protoc_insertion_point(class_scope:netaos.framework.proto.MlogConfig)
 private:
  class _Internal;

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
  static const ::PROTOBUF_NAMESPACE_ID::internal::LazyString _i_give_permission_to_break_this_code_default_log_level_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr log_level_;
  bool sw_;
  friend struct ::TableStruct_log_5fconf_2eproto;
};
// -------------------------------------------------------------------

class LogConfig PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:netaos.framework.proto.LogConfig) */ {
 public:
  inline LogConfig() : LogConfig(nullptr) {}
  virtual ~LogConfig();

  LogConfig(const LogConfig& from);
  LogConfig(LogConfig&& from) noexcept
    : LogConfig() {
    *this = ::std::move(from);
  }

  inline LogConfig& operator=(const LogConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline LogConfig& operator=(LogConfig&& from) noexcept {
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
  static const LogConfig& default_instance();

  static inline const LogConfig* internal_default_instance() {
    return reinterpret_cast<const LogConfig*>(
               &_LogConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(LogConfig& a, LogConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(LogConfig* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(LogConfig* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline LogConfig* New() const final {
    return CreateMaybeMessage<LogConfig>(nullptr);
  }

  LogConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<LogConfig>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const LogConfig& from);
  void MergeFrom(const LogConfig& from);
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
  void InternalSwap(LogConfig* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "netaos.framework.proto.LogConfig";
  }
  protected:
  explicit LogConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_log_5fconf_2eproto);
    return ::descriptor_table_log_5fconf_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kMlogConfigFieldNumber = 1,
  };
  // repeated .netaos.framework.proto.MlogConfig mlog_config = 1;
  int mlog_config_size() const;
  private:
  int _internal_mlog_config_size() const;
  public:
  void clear_mlog_config();
  ::netaos::framework::proto::MlogConfig* mutable_mlog_config(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::netaos::framework::proto::MlogConfig >*
      mutable_mlog_config();
  private:
  const ::netaos::framework::proto::MlogConfig& _internal_mlog_config(int index) const;
  ::netaos::framework::proto::MlogConfig* _internal_add_mlog_config();
  public:
  const ::netaos::framework::proto::MlogConfig& mlog_config(int index) const;
  ::netaos::framework::proto::MlogConfig* add_mlog_config();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::netaos::framework::proto::MlogConfig >&
      mlog_config() const;

  // @@protoc_insertion_point(class_scope:netaos.framework.proto.LogConfig)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::netaos::framework::proto::MlogConfig > mlog_config_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_log_5fconf_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// MlogConfig

// required string name = 1;
inline bool MlogConfig::_internal_has_name() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool MlogConfig::has_name() const {
  return _internal_has_name();
}
inline void MlogConfig::clear_name() {
  name_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& MlogConfig::name() const {
  // @@protoc_insertion_point(field_get:netaos.framework.proto.MlogConfig.name)
  return _internal_name();
}
inline void MlogConfig::set_name(const std::string& value) {
  _internal_set_name(value);
  // @@protoc_insertion_point(field_set:netaos.framework.proto.MlogConfig.name)
}
inline std::string* MlogConfig::mutable_name() {
  // @@protoc_insertion_point(field_mutable:netaos.framework.proto.MlogConfig.name)
  return _internal_mutable_name();
}
inline const std::string& MlogConfig::_internal_name() const {
  return name_.Get();
}
inline void MlogConfig::_internal_set_name(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline void MlogConfig::set_name(std::string&& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:netaos.framework.proto.MlogConfig.name)
}
inline void MlogConfig::set_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(value), GetArena());
  // @@protoc_insertion_point(field_set_char:netaos.framework.proto.MlogConfig.name)
}
inline void MlogConfig::set_name(const char* value,
    size_t size) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:netaos.framework.proto.MlogConfig.name)
}
inline std::string* MlogConfig::_internal_mutable_name() {
  _has_bits_[0] |= 0x00000001u;
  return name_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* MlogConfig::release_name() {
  // @@protoc_insertion_point(field_release:netaos.framework.proto.MlogConfig.name)
  if (!_internal_has_name()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return name_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void MlogConfig::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), name,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:netaos.framework.proto.MlogConfig.name)
}

// required string log_level = 2 [default = "info"];
inline bool MlogConfig::_internal_has_log_level() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool MlogConfig::has_log_level() const {
  return _internal_has_log_level();
}
inline void MlogConfig::clear_log_level() {
  log_level_.ClearToDefault(::netaos::framework::proto::MlogConfig::_i_give_permission_to_break_this_code_default_log_level_, GetArena());
  _has_bits_[0] &= ~0x00000002u;
}
inline const std::string& MlogConfig::log_level() const {
  // @@protoc_insertion_point(field_get:netaos.framework.proto.MlogConfig.log_level)
  if (log_level_.IsDefault(nullptr)) return _i_give_permission_to_break_this_code_default_log_level_.get();
  return _internal_log_level();
}
inline void MlogConfig::set_log_level(const std::string& value) {
  _internal_set_log_level(value);
  // @@protoc_insertion_point(field_set:netaos.framework.proto.MlogConfig.log_level)
}
inline std::string* MlogConfig::mutable_log_level() {
  // @@protoc_insertion_point(field_mutable:netaos.framework.proto.MlogConfig.log_level)
  return _internal_mutable_log_level();
}
inline const std::string& MlogConfig::_internal_log_level() const {
  return log_level_.Get();
}
inline void MlogConfig::_internal_set_log_level(const std::string& value) {
  _has_bits_[0] |= 0x00000002u;
  log_level_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, value, GetArena());
}
inline void MlogConfig::set_log_level(std::string&& value) {
  _has_bits_[0] |= 0x00000002u;
  log_level_.Set(
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:netaos.framework.proto.MlogConfig.log_level)
}
inline void MlogConfig::set_log_level(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000002u;
  log_level_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, ::std::string(value), GetArena());
  // @@protoc_insertion_point(field_set_char:netaos.framework.proto.MlogConfig.log_level)
}
inline void MlogConfig::set_log_level(const char* value,
    size_t size) {
  _has_bits_[0] |= 0x00000002u;
  log_level_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:netaos.framework.proto.MlogConfig.log_level)
}
inline std::string* MlogConfig::_internal_mutable_log_level() {
  _has_bits_[0] |= 0x00000002u;
  return log_level_.Mutable(::netaos::framework::proto::MlogConfig::_i_give_permission_to_break_this_code_default_log_level_, GetArena());
}
inline std::string* MlogConfig::release_log_level() {
  // @@protoc_insertion_point(field_release:netaos.framework.proto.MlogConfig.log_level)
  if (!_internal_has_log_level()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000002u;
  return log_level_.ReleaseNonDefault(nullptr, GetArena());
}
inline void MlogConfig::set_allocated_log_level(std::string* log_level) {
  if (log_level != nullptr) {
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  log_level_.SetAllocated(nullptr, log_level,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:netaos.framework.proto.MlogConfig.log_level)
}

// optional bool sw = 3 [default = true];
inline bool MlogConfig::_internal_has_sw() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool MlogConfig::has_sw() const {
  return _internal_has_sw();
}
inline void MlogConfig::clear_sw() {
  sw_ = true;
  _has_bits_[0] &= ~0x00000004u;
}
inline bool MlogConfig::_internal_sw() const {
  return sw_;
}
inline bool MlogConfig::sw() const {
  // @@protoc_insertion_point(field_get:netaos.framework.proto.MlogConfig.sw)
  return _internal_sw();
}
inline void MlogConfig::_internal_set_sw(bool value) {
  _has_bits_[0] |= 0x00000004u;
  sw_ = value;
}
inline void MlogConfig::set_sw(bool value) {
  _internal_set_sw(value);
  // @@protoc_insertion_point(field_set:netaos.framework.proto.MlogConfig.sw)
}

// -------------------------------------------------------------------

// LogConfig

// repeated .netaos.framework.proto.MlogConfig mlog_config = 1;
inline int LogConfig::_internal_mlog_config_size() const {
  return mlog_config_.size();
}
inline int LogConfig::mlog_config_size() const {
  return _internal_mlog_config_size();
}
inline void LogConfig::clear_mlog_config() {
  mlog_config_.Clear();
}
inline ::netaos::framework::proto::MlogConfig* LogConfig::mutable_mlog_config(int index) {
  // @@protoc_insertion_point(field_mutable:netaos.framework.proto.LogConfig.mlog_config)
  return mlog_config_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::netaos::framework::proto::MlogConfig >*
LogConfig::mutable_mlog_config() {
  // @@protoc_insertion_point(field_mutable_list:netaos.framework.proto.LogConfig.mlog_config)
  return &mlog_config_;
}
inline const ::netaos::framework::proto::MlogConfig& LogConfig::_internal_mlog_config(int index) const {
  return mlog_config_.Get(index);
}
inline const ::netaos::framework::proto::MlogConfig& LogConfig::mlog_config(int index) const {
  // @@protoc_insertion_point(field_get:netaos.framework.proto.LogConfig.mlog_config)
  return _internal_mlog_config(index);
}
inline ::netaos::framework::proto::MlogConfig* LogConfig::_internal_add_mlog_config() {
  return mlog_config_.Add();
}
inline ::netaos::framework::proto::MlogConfig* LogConfig::add_mlog_config() {
  // @@protoc_insertion_point(field_add:netaos.framework.proto.LogConfig.mlog_config)
  return _internal_add_mlog_config();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::netaos::framework::proto::MlogConfig >&
LogConfig::mlog_config() const {
  // @@protoc_insertion_point(field_list:netaos.framework.proto.LogConfig.mlog_config)
  return mlog_config_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace proto
}  // namespace framework
}  // namespace netaos

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_log_5fconf_2eproto
