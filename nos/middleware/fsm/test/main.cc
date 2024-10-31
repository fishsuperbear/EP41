
#include "bdb_wrapper.h"
#include "fsm_utils.h"

hozon::fsmcore::BdbWrapper _bdb{"fsmcoretest"};

bool bdb_add() {
  int speed_sign = 100;
  return (_bdb.set_basic("test_add", speed_sign, false));
}

bool bdb_del() {
  return (_bdb.del("test_add"));
}

bool test_bdb_add() {
  if (!bdb_add()) {
    std::cout << "test_bdb_add failed when add." << std::endl;
    return false;
  }

  if (!bdb_del()) {
    std::cout << "test_bdb_add failed when del." << std::endl;
    return false;
  }

  if (_bdb.has("test_add")) {
    std::cout << "deleted key but has this key yet." << std::endl;
    return false;
  }

  return true;
}

bool test_bdb_del_no_exist() {
  if (_bdb.has("test_bdb_del_no_exist")) {
    std::cout << "key test_bdb_del_no_exist should not has." << std::endl;
    return false;
  }

  if (_bdb.del("test_bdb_del_no_exist")) {
    std::cout << "delete no exist key test_bdb_del_no_exist should return ok." << std::endl;
    return true;
  }
  return false;
}

void test_db_sync() {
  int speed_sign2;
  if (_bdb.has("test_db_sync")) {
    _bdb.get_basic("test_db_sync", speed_sign2);
    std::cout << "key test_db_sync is already exist, value: " << speed_sign2 << std::endl;
  }
  int speed_sign = 100;
  _bdb.set_basic("test_db_sync", speed_sign, true);
  if (_bdb.has("test_db_sync")) {
    std::cout << "add key test_db_sync is exist." << std::endl;
  }
}

void test_get_no_exist() {
  int no_exist;
  auto ret = _bdb.get_basic("no_exist", no_exist);
  std::cout << "Get no exist return " << ret << std::endl;
}

int main() {
  if (test_bdb_add()) {
    std::cout << "test_bdb_add Passed!" << std::endl;
  }

  if (test_bdb_del_no_exist()) {
    std::cout << "test_bdb_del_no_exist passed!" << std::endl;
  }

  test_db_sync();

  test_get_no_exist();

  // 计算存 1万 次所需要时间，tsr 在车速大于 80km/h 时，需要发送特殊的can消息
  // 测试一下能不能每次 process 都存 db
  auto before = hozon::fsmcore::now_usec();
  int speed = 1;
  for (int ii = 0; ii < 10000; ++ii) {
    ++speed;
    _bdb.set_basic("speed_nosync", speed, false);
  }

  auto after = hozon::fsmcore::now_usec();
  std::cout << "save 1w times nosync used " << after - before << " micro seconds." << std::endl;

  before = hozon::fsmcore::now_usec();
  speed = 1;
  for (int ii = 0; ii < 10000; ++ii) {
    ++speed;
    _bdb.set_basic("speed_sync", speed, true);
  }

  after = hozon::fsmcore::now_usec();
  std::cout << "save 1w times sync used " << after - before << " micro seconds." << std::endl;

  return 0;
}