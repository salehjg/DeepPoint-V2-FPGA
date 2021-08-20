#pragma once

//  https://stackoverflow.com/a/12262626/8296604
//  USAGE:
//        throw std::runtime_error(Formatter() << foo << 13 << ", bar" << myData);
//        throw std::runtime_error(Formatter() << foo << 13 << ", bar" << myData >> Formatter::to_str);


#include <stdexcept>
#include <sstream>
#include <string>

class CStringFormatter {
 public:
  CStringFormatter() {}
  ~CStringFormatter() {}

  template<typename Type>
  CStringFormatter &operator<<(const Type &value) {
    stream_ << value;
    return *this;
  }

  std::string str() const { return stream_.str(); }
  operator std::string() const { return stream_.str(); }

  enum ConvertToString {
    to_str
  };
  std::string operator>>(ConvertToString) { return stream_.str(); }

 private:
  std::stringstream stream_;

  CStringFormatter(const CStringFormatter &);
  CStringFormatter &operator=(CStringFormatter &);
};


