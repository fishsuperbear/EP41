#pragma once

namespace {

struct AccessLogIt {
  typedef void (spdlog::logger::*fn)(const spdlog::details::log_msg&, bool,
                                     bool);
  friend fn get_fn(AccessLogIt);
};

template <typename Tag, typename Tag::fn M>
struct Rob {
  friend typename Tag::fn get_fn(Tag) { return M; }
};

}  // namespace

template struct Rob<AccessLogIt, &spdlog::logger::log_it_>;

namespace netaos {
namespace framework {
namespace loginterface {

class GroupLoggerBase {
 public:
  GroupLoggerBase(const std::string& name) : _name(name){};

  /*************************************************************************/
  /* log interface, derived class should rewrite                           */
  /*************************************************************************/
  virtual void flush() = 0;
  virtual void set_level(spdlog::level::level_enum level) = 0;
  virtual void log(const char* filename_in, int line_in,
                   netaos::framework::LogLevel level, const char* data,
                   size_t size) = 0;

  std::string get_name() const { return _name; }

 private:
  std::string _name;
};

}  // namespace loginterface
}  // namespace framework
}  // namespace netaos
