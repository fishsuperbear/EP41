/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_client_impl.cpp
 * Created on:
 * Author: jiangxiang
 *
 */
#ifndef MIDDLEWARE_DEVM_INCLUDE_COMMON_SINGLETON_H_
#define MIDDLEWARE_DEVM_INCLUDE_COMMON_SINGLETON_H_

#include <mutex>

template <typename T>
class Singleton {
    static T* c_instance;

   public:
    static T* GetInstance();
    ~Singleton();
};

template <typename T>
T* Singleton<T>::c_instance = NULL;
// T* Singleton<T>::c_instance = new T();

template <typename T>
T* Singleton<T>::GetInstance() {
    static std::once_flag flag;
    if (c_instance == NULL) {
        std::call_once(flag, [&] { c_instance = new T(); });
    }
    return c_instance;
}

template <typename T>
Singleton<T>::~Singleton() {
    if (c_instance != nullptr) {
        delete c_instance;
    }
}

#endif  // MIDDLEWARE_DEVM_INCLUDE_COMMON_SINGLETON_H_
