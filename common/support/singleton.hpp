#ifndef AVA_SINGLETON_HPP_
#define AVA_SINGLETON_HPP_

#include <memory>

namespace ava {
namespace support {

template <typename T>
class Singleton {
 public:
  static T *instance() {
    static const std::unique_ptr<T> instance{new T()};
    return instance.get();
  }

  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton) = delete;

 protected:
  Singleton() {}
};

}  // namespace support
}  // namespace ava

#endif  // AVA_SINGLETON_HPP_
