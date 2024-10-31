#include "bdb_wrapper.h"

#include "string_utils.h"

namespace hozon {
namespace fsmcore {

/*****************************************************************************/
/* construct                                                                 */
/*****************************************************************************/
BdbWrapper::BdbWrapper(const std::string& module) {
  _table = module + ".db";
  open(_table);
}

/*****************************************************************************/
/* destruct                                                                  */
/*****************************************************************************/
BdbWrapper::~BdbWrapper() { close(); }

/*****************************************************************************/
/* open, only do once                                                        */
/*****************************************************************************/
bool BdbWrapper::open(const std::string& file) {
  _no_sync_db = std::make_shared<Db>(
      nullptr, DB_CXX_NO_EXCEPTIONS);  // No exceptions, return errorno
  _sync_db = std::make_shared<Db>(
      nullptr, DB_CXX_NO_EXCEPTIONS);  // No exceptions, return errorno

  int32_t res = _no_sync_db->open(nullptr,   // Transaction pointer
                                  nullptr,   // No sync no need filename
                                  nullptr,   // Optional logical database name
                                  DB_BTREE,  // Database access method
                                  DB_CREATE | DB_THREAD,  // Open flags
                                  0);  // File mode (using defaults)

  if (0 != res) {
    std::cout << "Init berkeley db failed, return code: " << res << std::endl;
    return false;
  }

  res = _sync_db->open(nullptr,                // Transaction pointer
                       file.c_str(),           // Database file name
                       nullptr,                // Optional logical database name
                       DB_BTREE,               // Database access method
                       DB_CREATE | DB_THREAD,  // Open flags
                       0);                     // File mode (using defaults)

  if (0 != res) {
    std::cout << "Init berkeley db failed, file: " << file
              << ", return code: " << res << std::endl;
    return false;
  }

  return true;  // 0 as true
}

/*****************************************************************************/
/* close, only do once                                                       */
/*****************************************************************************/
bool BdbWrapper::close() {
  int32_t res = 0;
  if (_sync_db != nullptr) {
    res = _sync_db->close(0);
  }

  if (_no_sync_db != nullptr) {
    res = _no_sync_db->close(DB_NOSYNC);
  }

  return res == 0;  // 0 as true
}

/*****************************************************************************/
/* set for one string                                                        */
/*****************************************************************************/
bool BdbWrapper::set(const std::string& key, const std::string& value,
                     bool sync) {
  char _key[key.length() + 1];
  char _value[value.length() + 1];
  string_to_char(key, _key, key.length());
  string_to_char(value, _value, value.length());

  Dbt db_key(_key, key.length());
  Dbt db_value(_value, value.length());

  // 读全局 map 确保线程安全
  std::lock_guard<std::mutex> lg(_sync_map.mtx);
  if (_sync_map.data_map.find(key) == _sync_map.data_map.end()) {
    _sync_map.data_map[key] = sync;
  }

  // 需要写磁盘
  if (_sync_map.data_map[key]) {
    auto ret = _sync_db->put(nullptr, &db_key, &db_value, 0);
    if (ret != 0) {
      std::cout << "Put to sync db failed, key: " << key << ", value: " << value
                << ", return: " << ret << std::endl;
      return false;
    }
    _sync_db->sync(0);
  } else {
    auto ret = _no_sync_db->put(nullptr, &db_key, &db_value, 0);
    if (ret != 0) {
      std::cout << "Put to no sync db failed, key: " << key
                << ", value: " << value << ", return: " << ret << std::endl;
      return false;
    }
  }

  return true;
}

/*****************************************************************************/
/* get for one string                                                        */
/*****************************************************************************/
bool BdbWrapper::get(const std::string& key, std::string& value) {
  char _key[key.length() + 1];
  string_to_char(key, _key, key.length());
  Dbt db_key(_key, key.length());

  // try first
  char try_buffer[BDB_BUFFER_SIZE];
  memset(try_buffer, 0, BDB_BUFFER_SIZE);

  Dbt db_value;
  db_value.set_data(try_buffer);
  db_value.set_ulen(BDB_BUFFER_SIZE);
  db_value.set_flags(DB_DBT_USERMEM);  // for db open flags DB_THREAD

  // 优先去磁盘中查找
  int32_t res = _sync_db->get(nullptr,    // Transaction pointer
                              &db_key,    // Key
                              &db_value,  // Value
                              0);         // Get flags (using defaults)
  // 能查到
  if (res == 0) {
    value.assign(static_cast<char*>(db_value.get_data()), db_value.get_size());
    std::lock_guard<std::mutex> lg(_sync_map.mtx);
    _sync_map.data_map[key] = true;

    return true;
  }

  if (res == DB_BUFFER_SMALL) {
    int32_t needed_size = db_value.get_size();
    char needed_buffer[needed_size];

    // get again
    db_value.set_data(needed_buffer);
    db_value.set_ulen(needed_size);
    db_value.set_flags(DB_DBT_USERMEM);  // for db open flags DB_THREAD

    int32_t res = _sync_db->get(nullptr,    // Transaction pointer
                                &db_key,    // Key
                                &db_value,  // Value
                                0);         // Get flags (using defaults)
    if (res == 0) {
      value.assign(static_cast<char*>(db_value.get_data()),
                   db_value.get_size());

      std::lock_guard<std::mutex> lg(_sync_map.mtx);
      _sync_map.data_map[key] = true;

      return true;
    } else {
      std::cout << "extend buffer size also get value failed, return: " << res
                << std::endl;
      return false;
    }
  }

  // 查不到，再去内存中查找
  res = _no_sync_db->get(nullptr,    // Transaction pointer
                         &db_key,    // Key
                         &db_value,  // Value
                         0);         // Get flags (using defaults)
  // 能查到
  if (res == 0) {
    value.assign(static_cast<char*>(db_value.get_data()), db_value.get_size());
    std::lock_guard<std::mutex> lg(_sync_map.mtx);
    _sync_map.data_map[key] = false;

    return true;
  }

  if (res == DB_BUFFER_SMALL) {
    int32_t needed_size = db_value.get_size();
    char needed_buffer[needed_size];

    // get again
    db_value.set_data(needed_buffer);
    db_value.set_ulen(needed_size);
    db_value.set_flags(DB_DBT_USERMEM);  // for db open flags DB_THREAD

    int32_t res = _sync_db->get(nullptr,    // Transaction pointer
                                &db_key,    // Key
                                &db_value,  // Value
                                0);         // Get flags (using defaults)
    if (res == 0) {
      value.assign(static_cast<char*>(db_value.get_data()),
                   db_value.get_size());

      std::lock_guard<std::mutex> lg(_sync_map.mtx);
      _sync_map.data_map[key] = false;

      return true;
    } else {
      std::cout << "extend buffer size also get value failed, return: " << res
                << std::endl;
      return false;
    }
  }

  return false;
}

/*****************************************************************************/
/* judge whether key is existed                                              */
/*****************************************************************************/
bool BdbWrapper::has(const std::string& key) {
  char _key[key.length() + 1];
  string_to_char(key, _key, key.length());
  Dbt db_key(_key, key.length());

  // 优先去磁盘中查找
  int32_t res = _sync_db->exists(nullptr,  // Transaction pointer
                                 &db_key,  // Key
                                 0);       // Check flags (using defaults)

  if (res == DB_NOTFOUND) {
    // 磁盘找不到，再去内存查找
    int32_t res = _no_sync_db->exists(nullptr,  // Transaction pointer
                                      &db_key,  // Key
                                      0);       // Check flags (using defaults)
    if (res == DB_NOTFOUND) {
      return false;
    }
  }

  return true;
}

/*****************************************************************************/
/* deleted key and releated value                                            */
/*****************************************************************************/
bool BdbWrapper::del(const std::string& key) {
  char _key[key.length() + 1];
  string_to_char(key, _key, key.length());
  Dbt db_key(_key, key.length());

  // 优先去磁盘中查找
  int32_t res = _sync_db->exists(nullptr,  // Transaction pointer
                                 &db_key,  // Key
                                 0);       // Check flags (using defaults)
  if (res == DB_NOTFOUND) {
    // 磁盘找不到，再去内存查找
    res = _no_sync_db->exists(nullptr,  // Transaction pointer
                              &db_key,  // Key
                              0);       // Check flags (using defaults)
    if (res == DB_NOTFOUND) {
      return true;
    } else {
      res = _no_sync_db->del(nullptr,  // Transaction pointer
                             &db_key,  // Key
                             0);       // Deelete flags (using defaults)
      if (0 != res) {
        std::cout << "delete " << key
                  << " from nosync bdb failed, return code: " << res
                  << std::endl;
        return false;
      } else {
        std::cout << "delete " << key << " from nosync bdb ok." << std::endl;
        return true;
      }
    }
  } else {
    res = _sync_db->del(nullptr,  // Transaction pointer
                        &db_key,  // Key
                        0);       // Deelete flags (using defaults)

    if (0 != res) {
      std::cout << "delete " << key
                << " from sync bdb failed, return code: " << res << std::endl;
      return false;
    } else {
      std::cout << "delete " << key << " from sync bdb ok." << std::endl;
      _sync_db->sync(0);
      return true;
    }
  }
}

}  // namespace fsmcore
}  // namespace hozon