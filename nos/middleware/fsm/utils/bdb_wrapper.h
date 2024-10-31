#pragma once

#include <db_cxx.h>

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include "string_utils.h"

namespace hozon {
namespace fsmcore {

class BdbWrapper {
 public:
  BdbWrapper(const std::string& module);
  ~BdbWrapper();

  bool set(const std::string& key, const std::string& value, bool sync = true);
  bool set(const std::string& key, const std::vector<std::string>& value,
           bool sync = true);
  template <typename T>
  bool set_basic(const std::string& key, const T value, bool sync = true);

  bool get(const std::string& key, std::string& value);
  bool get(const std::string& key, std::vector<std::string>& value);
  template <typename T>
  bool get_basic(const std::string& key, T& value);

  bool has(const std::string& key);
  bool del(const std::string& key);

 private:
  // noncopyable
  BdbWrapper(const BdbWrapper&) = delete;
  BdbWrapper& operator=(const BdbWrapper&) = delete;

  bool open(const std::string& file);
  bool close();

  std::string _table;
  std::shared_ptr<Db> _no_sync_db;
  std::shared_ptr<Db> _sync_db;
  struct threadsafe_unordered_map {
    std::unordered_map<std::string, bool> data_map;
    std::mutex mtx;
  };
  threadsafe_unordered_map _sync_map;
  const uint32_t BDB_BUFFER_SIZE{1024};
};

/*****************************************************************************/
/* set with template                                                         */
/*****************************************************************************/
template <typename T>
bool BdbWrapper::set_basic(const std::string& key, const T value, bool sync) {
  char _key[key.length() + 1];
  string_to_char(key, _key, key.length());

  Dbt db_key(_key, key.length());
  Dbt db_value((void*)&value, sizeof(value));

  // 读全局 map 确保线程安全
  std::lock_guard<std::mutex> lg(_sync_map.mtx);
  if (_sync_map.data_map.find(key) == _sync_map.data_map.end()) {
    _sync_map.data_map[key] = sync;
  }

  // 需要写磁盘
  if (_sync_map.data_map[key]) {
    auto ret = _sync_db->put(nullptr, &db_key, &db_value, 0);
    if (ret != 0) {
      std::cout << "Put to sync db failed, key: " << key << ", value: " << value << ", return: " << ret << std::endl;
      return false;
    }
    _sync_db->sync(0);
  } else {
    auto ret = _no_sync_db->put(nullptr, &db_key, &db_value, 0);
    if (ret != 0) {
      std::cout << "Put to no sync db failed, key: " << key << ", value: " << value << ", return: " << ret << std::endl;
      return false;
    }
  }

  return true;
}

/*****************************************************************************/
/* get with template                                                         */
/*****************************************************************************/
template <typename T>
bool BdbWrapper::get_basic(const std::string& key, T& value) {
  char _key[key.length() + 1];
  string_to_char(key, _key, key.length());
  Dbt db_key(_key, key.length());

  // try first
  T try_buffer;

  Dbt db_value;
  db_value.set_data((void*)&try_buffer);
  db_value.set_ulen(sizeof(try_buffer));
  db_value.set_flags(DB_DBT_USERMEM);  // for db open flags DB_THREAD

  // 优先去磁盘中查找
  int32_t res = _sync_db->get(nullptr,    // Transaction pointer
                              &db_key,    // Key
                              &db_value,  // Value
                              0);         // Get flags (using defaults)
  // 能查到
  if (res == 0) {
    value = try_buffer;
    std::lock_guard<std::mutex> lg(_sync_map.mtx);
    _sync_map.data_map[key] = true;

    return true;
  }

  // 查不到，再去内存中查找
  res = _no_sync_db->get(nullptr,    // Transaction pointer
                        &db_key,    // Key
                        &db_value,  // Value
                        0);         // Get flags (using defaults)
  // 能查到
  if (res == 0) {
    value = try_buffer;
    std::lock_guard<std::mutex> lg(_sync_map.mtx);
    _sync_map.data_map[key] = false;

    return true;
  }

  return false;
}

}  // namespace fsmcore
}  // namespace hozon