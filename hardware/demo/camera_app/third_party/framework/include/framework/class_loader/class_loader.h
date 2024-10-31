#ifndef CYBER_CLASS_LOADER_CLASS_LOADER_H_
#define CYBER_CLASS_LOADER_CLASS_LOADER_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "framework/class_loader/class_loader_register_macro.h"

namespace netaos {
namespace framework {
namespace class_loader {

/**
 *  for library load,createclass object
 */
class ClassLoader {
 public:
  explicit ClassLoader(const std::string& library_path);
  virtual ~ClassLoader();

  bool IsLibraryLoaded();
  bool LoadLibrary();
  int UnloadLibrary();
  const std::string GetLibraryPath() const;
  template <typename Base>
  std::vector<std::string> GetValidClassNames();
  template <typename Base>
  std::shared_ptr<Base> CreateClassObj(const std::string& class_name);
  template <typename Base>
  bool IsClassValid(const std::string& class_name);

 private:
  template <typename Base>
  void OnClassObjDeleter(Base* obj);

 private:
  std::string library_path_;
  int loadlib_ref_count_;
  std::mutex loadlib_ref_count_mutex_;
  int classobj_ref_count_;
  std::mutex classobj_ref_count_mutex_;
};

template <typename Base>
std::vector<std::string> ClassLoader::GetValidClassNames() {
  return (utility::GetValidClassNames<Base>(this));
}

template <typename Base>
bool ClassLoader::IsClassValid(const std::string& class_name) {
  std::vector<std::string> valid_classes = GetValidClassNames<Base>();
  return (std::find(valid_classes.begin(), valid_classes.end(), class_name) !=
          valid_classes.end());
}

template <typename Base>
std::shared_ptr<Base> ClassLoader::CreateClassObj(
    const std::string& class_name) {
  if (!IsLibraryLoaded()) {
    LoadLibrary();
  }

  Base* class_object = utility::CreateClassObj<Base>(class_name, this);
  if (class_object == nullptr) {
    AWARN << "CreateClassObj failed, ensure class has been registered. "
          << "classname: " << class_name << ",lib: " << GetLibraryPath();
    return std::shared_ptr<Base>();
  }

  std::lock_guard<std::mutex> lck(classobj_ref_count_mutex_);
  classobj_ref_count_ = classobj_ref_count_ + 1;
  std::shared_ptr<Base> classObjSharePtr(
      class_object, std::bind(&ClassLoader::OnClassObjDeleter<Base>, this,
                              std::placeholders::_1));
  return classObjSharePtr;
}

template <typename Base>
void ClassLoader::OnClassObjDeleter(Base* obj) {
  if (nullptr == obj) {
    return;
  }

  std::lock_guard<std::mutex> lck(classobj_ref_count_mutex_);
  delete obj;
  --classobj_ref_count_;
}

}  // namespace class_loader
}  // namespace framework
}  // namespace netaos
#endif  // CYBER_CLASS_LOADER_CLASS_LOADER_H_
