#ifndef AVA_SINGLETON_HPP_
#define AVA_SINGLETON_HPP_

#include <memory>

template <typename T>
class Singleton {
 public:
  static T &instance() {
    static const std::unique_ptr<T> instance{new T()};
    return *instance;
  }

  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton) = delete;

 protected:
  Singleton() {}
};

class ApiServerSetting final : public Singleton<ApiServerSetting> {
 public:
  void set_listen_port(unsigned int p) { listen_port = p; }

  unsigned int get_listen_port() const { return listen_port; }

 private:
  unsigned int listen_port;
};

#endif  // AVA_SINGLETON_HPP_
