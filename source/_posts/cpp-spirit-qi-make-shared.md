---
title: 在Spirit Qi中使用std::shared_ptr对象
date: 2024-12-09 22:23:29
tags: ["C++"]
---

`boost::spirit::qi`是一个用于解析字符串的库，可以用于解析字符串并构造对象。在解析的过程中，有时候需要构造语法树，这个时候通常可以采用`phx::new_`或者`phx::construct`来构造裸指针或对象。但是在实际的应用当中，有时候会希望通过如`std::shared_ptr`这样的智能指针来管理对象的生命周期，但是在`boost::phoenix`当中并没有提供`make_shared`这样的函数对象。

基于[stackoverflow](https://stackoverflow.com/questions/30763212/making-a-vector-of-shared-pointers-from-spirit-qi)上的回答可以构造一个`make_shared_`的函数对象，用于在`boost::spirit::qi`的语法规则当中延迟构造`std::shared_ptr`对象。

```cpp
namespace boost::phoenix {
template <typename T>
struct make_shared_f {
  template <typename... A>
  struct result {
    typedef std::shared_ptr<T> type;
  };

  template <typename... A>
  typename result<A...>::type operator()(A&&... a) const {
    return std::make_shared<T>(std::forward<A>(a)...);
  }
};
template <typename T>
using make_shared_ = function<make_shared_f<T>>;
}  // namespace boost::phoenix
```

在使用的时候可以如下构造创建`std::shared_ptr`对象的action：

```cpp
namespace qi = boost::spirit::qi;
namespace phx = boost::phoenix;

using phx::make_shared_;
using qi::_1;
using qi::_val;

...

expr = term[_val = _1] >>
       *(('+' >> term[_val = make_shared_<AddNode>()(_val, _1)]) |
         ('-' >> term[_val = make_shared_<SubNode>()(_val, _1)]));
```