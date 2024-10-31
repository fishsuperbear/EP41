#ifndef CYBER_CLASS_LOADER_UTILITY_CLASS_FACTORY_H_
#define CYBER_CLASS_LOADER_UTILITY_CLASS_FACTORY_H_

#include <string>
#include <typeinfo>
#include <vector>

namespace netaos {
namespace framework {
namespace class_loader {

class ClassLoader;

namespace utility {

class AbstractClassFactoryBase {
 public:
  AbstractClassFactoryBase(const std::string& class_name,
                           const std::string& base_class_name);
  virtual ~AbstractClassFactoryBase();

  void SetRelativeLibraryPath(const std::string& library_path);
  void AddOwnedClassLoader(ClassLoader* loader);
  void RemoveOwnedClassLoader(const ClassLoader* loader);
  bool IsOwnedBy(const ClassLoader* loader);
  bool IsOwnedByAnybody();
  std::vector<ClassLoader*> GetRelativeClassLoaders();
  const std::string GetRelativeLibraryPath() const;
  const std::string GetBaseClassName() const;
  const std::string GetClassName() const;

 protected:
  std::vector<ClassLoader*> relative_class_loaders_;
  std::string relative_library_path_;
  std::string base_class_name_;
  std::string class_name_;
};

template <typename Base>
class AbstractClassFactory : public AbstractClassFactoryBase {
 public:
  AbstractClassFactory(const std::string& class_name,
                       const std::string& base_class_name)
      : AbstractClassFactoryBase(class_name, base_class_name) {}

  virtual Base* CreateObj() const = 0;

 private:
  AbstractClassFactory();
  AbstractClassFactory(const AbstractClassFactory&);
  AbstractClassFactory& operator=(const AbstractClassFactory&);
};

template <typename ClassObject, typename Base>
class ClassFactory : public AbstractClassFactory<Base> {
 public:
  ClassFactory(const std::string& class_name,
               const std::string& base_class_name)
      : AbstractClassFactory<Base>(class_name, base_class_name) {}

  Base* CreateObj() const { return new ClassObject; }
};

}  // namespace utility
}  // namespace class_loader
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_CLASS_LOADER_UTILITY_CLASS_FACTORY_H_
