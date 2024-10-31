/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_macros.h
 * @Date: 2023/07/24
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_COMMON_DC_MACROS_H_
#define MIDDLEWARE_TOOLS_COMMON_DC_MACROS_H_

#include <iostream>
#include <sstream>
#include <map>
#include <mutex>

#ifndef _DEBUG
#define DEBUG_VAL(value) do { printf("%s:%d @%s | ", \
	__FILE__, __LINE__ , __FUNCTION__);           \
    cout<<#value<<"="<<value<<endl; \
} while(0);

#define DEBUG_LOG(format, ...) do {  fprintf(stderr, "%s:%d @%s | " format "\n", \
        __FILE__, __LINE__ , __FUNCTION__, ##__VA_ARGS__ ); \
    } while(0);
#else
#define DEBUG_VAL(...) do {} while(0);
#define DEBUG_LOG(...) do {} while(0);
#endif



//void DEBUG_PRINT_MAIN(const char* sep) {
//}
//template<typename Arg1, typename... Argn>
//void DEBUG_PRINT_MAIN(const char* sep,const Arg1 &arg1, const Argn &... args) {
//  std::cout<<arg1<<sep;
//  DEBUG_PRINT_MAIN(sep, args...);
//}
//template<typename Arg1, typename... Argn>
//void DEBUG_PRINT(const Arg1 &arg1, const Argn &... args) {
//  std::stringstream line;
//  line<<__LINE__;
//  DEBUG_PRINT_MAIN("",  __FILE__, ":", __LINE__ ," ");
//  DEBUG_PRINT_MAIN(" ",arg1, args...);
//  std::cout<<""<<std::endl;
//}

class PointerManager {
 public:
  static PointerManager *getInstance() {
    static PointerManager pm;
    return &pm;
  }
  void add(void *position, std::string infomation) {
    std::lock_guard<std::mutex> lg(mtx_);
    rec[position] = infomation;
  }
  void del(void * position, std::string infomation) {
    std::lock_guard<std::mutex> lg(mtx_);
    if (rec.find(position)==rec.end()) {
        std::cout<<"==="<<infomation<<" delete info not found:"<<std::endl;
        return;
    }
    rec.erase(position);
  }
 private:
  PointerManager() {}
  PointerManager(PointerManager &&cf) = delete;
  ~PointerManager() {
    std::cout<<std::endl<<std::endl;
    std::cout<<"=============Pointer Manager destruct==============="<<std::endl;
    std::lock_guard<std::mutex> lg(mtx_);
    for (const auto& item: rec) {
      std::cout<<item.first<<" -- "<<item.second<<"; // was not delete in memory\n";
    }
    std::cout<<"=============Pointer Manager destruct end ==============="<<std::endl;
  }
  PointerManager(const PointerManager &cf) = delete;
  PointerManager &operator=(const PointerManager &cf) = delete;
  std::map<void *, std::string> rec;
  std::mutex mtx_;
};
#define DC_NEW(var, classname) \
    do {                       \
        var = new(std::nothrow) classname; \
        std::stringstream ss; ss<<__FILE__<<":"<<__LINE__<<" "<<#var <<"=new "<<#classname;                     \
        PointerManager::getInstance()->add((void *)var, ss.str())      ;           \
    } while (0);


#define DC_RETURN_NEW(classname) \
    do { \
        auto *temp = new(std::nothrow) classname; \
        std::stringstream ss; ss<<__FILE__<<":"<<__LINE__<<" return new "<<#classname;                     \
        PointerManager::getInstance()->add((void *)temp, ss.str())      ;           \
        return temp;                              \
    } while (0);

/**
 * Try delete the pointer p, and set p to nullptr.
 */
#define DC_DELETE(p) \
    do { \
        if (p ==  nullptr) { \
            break; \
        }            \
        std::stringstream ss; ss<<__FILE__<<":"<<__LINE__<<" " <<"=delete "<<p;\
PointerManager::getInstance()->del((void *)p,ss.str()) ;            \
        delete p; \
        p = nullptr; \
    } while (0);

/**
 * Try delete the pointer p[], and set p to nullptr.
 */
#define DC_DELETE_ARRAY(p) \
    do { \
        if (p == nullptr) { \
            break; \
        } \
        delete[] p; \
        PointerManager::getInstance()->del((void *)p);\
p = nullptr; \
    } while (0);


#endif  // MIDDLEWARE_TOOLS_COMMON_DC_MACROS_H_
