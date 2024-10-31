/********************************************************
 * Copyright (C) 2021 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 * Author: LiangChengpeng
 ********************************************************/
#pragma once
#include <fstream>
#include <iostream>
#include <string>

#include "yaml-cpp/yaml.h"

namespace hozon {
namespace fsmcore {

using YamlIterator = YAML::const_iterator;

class YamlNode {
 public:
  explicit YamlNode(const std::string path) {
    std::ifstream fin(path);
    if (!fin) {
      std::clog << "Error: File corrupted or not exist.: " << path << std::endl;
    } else {
      fin.close();
    }
    node = YAML::LoadFile(path);
  }

  YamlNode() {}

  explicit YamlNode(const YAML::Node child) { this->node = child; }
  ~YamlNode() {}

  // indexing
  template <typename Key>
  const YamlNode operator[](const Key &keyName) const {
    return YamlNode(node[keyName]);
  }

  template <typename Key>
  YamlNode operator[](const Key &keyName) {
    return YamlNode(node[keyName]);
  }

  // access
  template <typename T>
  T as() const {
    return node.as<T>();
  }

  template <typename T>
  bool GetValue(const std::string &keyName, T &value) const {  // NOLINT
    if (node[keyName]) {
      value = node[keyName].as<T>();
      return true;
    }
    return false;
  }

  YamlIterator begin() const { return node.begin(); }

  YamlIterator end() const { return node.end(); }

  bool IsSequence() const { return node.IsSequence(); }

  size_t NodeSize() const { return node.size(); }

  template <typename Key, typename Value>
  void SetNodeValue(const Key &keyName, const Value &value) {
    node[keyName] = value;
    return;
  }

  YAML::Node GetNode() const { return node; }

  bool HasKeyValue(const std::string &keyName) const {
    if (node[keyName]) {
      return true;
    }
    return false;
  }

 private:
  YAML::Node node;
};

}  // namespace fsmcore
}  // namespace hozon
