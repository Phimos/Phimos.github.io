---
title: "C++ 面试题 2"
date: 2020-02-06 06:14:51
tags: ["C++", "Interview"]
---

准备面试的时候找到了两套C++面试题，做了一下，解答直接写在上面了，放在这里分享出来。感觉难度并不是很高，这是第二套~

---

## A. Show me the code

#### 1. Please create a "String" class to process char strings. This string class should satisfy the following requirements:

- It could be used as an original type like "int" to define variables and support assignment and copy.
- It could be used as the parameter type of a function and the return value type of a function.
- It could be used as the element type of the STL container, e.g., vector/list/deque.

In other words, your "String" class could be used in the following code and be complied successfully by a standard C++ compiler.

```c++
void foo(String x)
{
}
void bar(const String& x)
{
}
String baz()
{
  String ret("world");
  return ret;
}
int main()
{
  String s0;
  String s1("hello");
  String s2(s0);
  String s3 = s1;
  s2 = s1;
  foo(s1);
  bar(s1);
  foo("temporary");
  bar("temporary");
  String s4 = baz();
  std::vector<String> svec;
  svec.push_back(s0);
  svec.push_back(s1);
  svec.push_back(baz());
  svec.push_back("good job");
}
```

略，感觉以前写过。



#### 2. Imagine we have a single linked list, please write a function which could remove a specified node in the list.

- Please define the data structure for the list as you need.
- Pass the head pointer to the function.
- Pass the pointer to the node to be removed to the function. Remove the node in the list if it has the same pointer value.

```c++
struct ListNode{
  int val;
  ListNode* next;
  ListNode(int _val): val(_val){}
};

ListNode* deleteNode(ListNode* head, int val)
{
  if(!head)
    return NULL;
  while(head->val == val)
  {
    head = head->next;
  }
  if(!head)
    return NULL;
  ListNode* pnode = head->next;
  ListNode* prev = head;
  while(pnode)
  {
    if(pnode->val == val)
    {
      prev->next = pnode->next;
      pnode = pnode->next;
    }
    else
    {
      prev = pnode;
      pnode = pnode->next;
    }
  }
  return head;
}
```



------

## B. Basic Part

#### 1. What are wrong in the following code? Please point out.

```c++
void main()
{
  char a = 'a';
  int b = 0;
  int *pInt1 = &b;
  int c = *pInt1;
  pInt1 = (int*)(&a);
  int *pInt2 = pInt1 + 1;
  int d = *pInt2;
  void *pV = &a;
  // char * pV = &a;
  pV++;	// 对空指针不能最算术运算
  char e = *pV;
}

```



#### 2. What are wrong in the following code? Please provide your **FIX**.

*Common.h*

```c++
int var1;
void foo(int input)
{
  // some code
}
```

*TestA.h*

```c++
#include "Common.h"
#include "TestB.h"

class CTestA
{
  private:
    CTestB m_b;
};
```

*TestB.h*

```c++
#include "Common.h"
#include "TestA.h"

class CTestB
{
  private:
    CTestA m_a;
};
```

*TestA.cpp*

```c++
#include "TestA.h"
// some code
```

*TestB.cpp*

```c++
#include "TestB.h"
// some code
```

提前声明，结构体内部都采用指针而不是实体。

---



## C. STL

#### 1. Errors, inefficiency and potential bugs exsit in the following code, please point them out.

```c++
int foo(std::map<int, int>& mapIn, std::set<int>& setIn)
{
  std::vector<int> va(10);
  std::vector<int> vb;
  std::copy(va.begin(), va.end(), vb.begin());
  
  //std::vector<int> vb(va);
  
  std::vector<int> vc(100);
  auto iter = va.begin() + 5;
  int varInt = *iter;
  va.push_back(vc.begin(), vc.end());
  varInt = *(++iter);
  if (mapIn[4] == 0)
  {
    // do something 
  }
  auto itVec = std::find(vc.begin(), vc.end(), 100);
  if (itVec != vc.end())
  {
    // do something
  }
  
  //auto itSet = setIn.find(10);
  //Set本来就是有序的结构，利用可以进行二分查找，效率更高
  
  auto itSet = std::find(setIn.begin(), setIn.end(), 10);
  if (itSet == setIn.end())
  {
    // do something
  }
}
```



#### 2. Please see the following code, `TypeA` could be either a function pointer or a functor, please try to provide the definition for `TypeA` in both function pointer way and functor way.

```c++
void foo(TypeA processor)
{
  int paraInt = 0;
  const char* pParaStr = "Test";
  int rtn = processor(paraInt, pParaStr);
}
```

函数指针:

```c++
int function_pointer(int pInt, const char* pStr)
{
		//do something
  	return 0;
} 
```

函数对象：

```c++
class functor{
public:
    int operator ()(int pInt, const char * pStr){
				// do something
        return 0;
    }  
};
```

