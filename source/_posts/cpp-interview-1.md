---
title: "C++ 面试题 1"
date: 2020-02-06 06:12:27
tags: ["C++", "Interview"]
---

准备面试的时候找到了两套C++面试题，做了一下，解答直接写在上面了，放在这里分享出来。感觉难度并不是很高，这是第一套~

---

## For your answers, either Chinese or English are OK.

## A. Coding

#### 1. Please create a class to represent Int. Only declaration is required. Please give the declaration of all member functions, friend functions, and member variables.

略



#### 2. Imagine we have a single linked list, please write a function which could move a specified node to the tail of the list.

- Please define the data structure for the list as you need.
- Pass the head pointer to the function.
- Pass the pointer of the node to be moved to the function. Move the node to the tail of the list if it has the same pointer value.



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



## B. Read and Answer

#### 1. What are wrong in the following code?

```c++
void foo(short startIndex)
{  
  short buffer[50000] = { 0 };
  for (auto i = startIndex; i < 40000; ++i) 
  {  
    buffer[i] = (char)i;  
  }
}
```

Because i is short, i < 40000  is always True, the loop NEVER ends.



#### 2. Mark all lines which are wrong and provide the FIX.

```c++
class CBase
{
public:
    CBase() 
      : m_pVar1(NULL)
    {
        m_pVar2 = new int(4);
    }
    ~CBase()
    {
        delete m_pVar2;
        delete m_pVar1;	// Can't delete NULL pointer
      /*
      *if(! m_pVar1)
      *  delete m_pVal1;
      */
    }
    void Init(int* pVar)
    {
        m_pVar1 = pVar;
    }
private:
    int* m_pVar1;
    int* m_pVar2;
}
class CDerive : public CBase
{
public:
    CDerive(int var) : m_var(var) {};
  //CDerive(){}
    ~CDerive() {};
private:
    int m_var;
}
int main()
{
    CDerive* pDerives = new CDerive[10]; // can't init
    int *pVar = new int(10);
    for (int i = 0; i < 10; ++i)
    {
        pDerive[i].Init(pVar);
    }
    delete pDerives;	// pDerives is an array
  //delete[] pDerives;
    delete pVar;
}
```



#### 3. The following code could not be compiled, give TWO ways to fix it.

```c++
class CFoo
{
public:
    CFoo() : m_var(0) {};
    ~CFoo() {};
    bool AddAndCheck() const
    {
        m_var += 5;
        return m_var < 10;
    }
private:
    int m_var;
};
```

第一种：丢掉const

```c++
class CFoo
{
public:
    CFoo() : m_var(0) {};
    ~CFoo() {};
    bool AddAndCheck()// drop the const
    {
        m_var += 5;
        return m_var < 10;
    }
private:
    int m_var;
};
```

第二种：加上mutable

```c++
class CFoo
{
public:
    CFoo() : m_var(0) {};
    ~CFoo() {};
    bool AddAndCheck() const
    {
        m_var += 5;
        return m_var < 10;
    }
private:
    mutable int m_var;
};
```



#### 4. What are wrong in the following code? Provide your fix.

```c++
#define INCREASE(a) ++a 
// #define INCREASE(a) a+1
void foo()
{
    int a = 3;
    int b = 7;
    cout<<INCREASE(a * b)<<endl; // 等价于(++a) * b
}
```



#### 5. What are wrong in the following code? Why?

```c++
char* GetStaticBuffer()
{
    char buffer[100] = { 0 };
    return buffer;
}
```

buffer在栈上分配的，返回之后释放了。



#### 6. With default configuration, in the following code, function fooA is OK but fooB has issues, why?

```c++
const int TEN_MEGA_BYTES = 10 * 1024 * 1024;
void fooA()
{
    char *pBuffer = new char[TEN_MEGA_BYTES];
}
void fooB()
{
    char buffer[TEN_MEGA_BYTES] = { 0 };
}
```

`fooA()`在堆上进行分配，`fooB()`在栈上进行分配，栈小，会爆。



#### 7. In the following code, line 32, I want to give "student_teacher" as the input parameter, but by mistake, I typed "student". The compiling succeeded anyway. Why? If I want the compiling be failed in this condition, how to do?

```c++
class CStudent
{
public:
    CStudent() {};
    ~CStudent() {};
};

class CTeacher
{
public:
    CTeacher() {};
  // delete this part------------
    CTeacher(CStudent student)
        : m_student(student)
    {
    }
    ~CTeacher() {};
  // end here -------------------
    void Teach() {};
private:
    CStudent m_student;
};

void foo(CTeacher teacher)
{
    teacher.Teach();
}
int main()
{
    CStudent student;
    CTeacher student_teacher;
    foo(student);
}
```



#### 8. What is the output of the following code?

```c++
int fooVar = 10;
void foo(int *pVar, int& var)
{
    pVar = &fooVar;
    var = fooVar;
}
int main()
{
    int var1 = 1;
    int var2 = 2;
    foo(&var1, var2);
    cout<<var1<<":"<<var2<<endl;
}
```

1:10



#### 9. How to run some code before main function is called?

```c++
int code_before_main()
{
    cout<<"code before main()"<<endl;
    return 0;
}

int useless = code_before_main();

int main()
{
    cout<<"main() start"<<endl;
}

```



#### 10. What the difference are between

std::vector::resize() and std::vector::reserve()

std::vector::size() and std::vector::capacity()

`size()`对应的是有效空间，而`capacity()`对应的是实际空间。

`resize()`调整的是有效空间的大小，`reserve()`调整的是实际空间大小。



#### 11. Template greater<> is often used when operating the STL containers. Please give an implementation of greater<>.

Hints: Two template parameters. One is a number to be compared as base, another is the element to be compared.

```c++
template <class T> 
struct greater{
  bool operator() (const T& x, const T& y) const 
  	{return x>y;}
};
```



#### 12. Say we have a class named Printer which has a member function as Print(). Now there's a container as vector. Please give a lambda expression which could be used in for_each() to call Print() function for every instance in vector.

```c++
for_each(v.begin(), v.end(), [](const T& n) { Printer().Print(n); });
```

