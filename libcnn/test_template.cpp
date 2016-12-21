
#include <iostream>

// explicit_specialization.cpp  
template<class T> void f(T t)  
{  
//std::cout << "default" << std::endl;
};  
  
// Explicit specialization of f with 'char' with the  
// template argument explicitly specified:  
//  
template<> void f<char>(char &c)  
{
 c++; 
//std::cout << "char" << std::endl;
}  
  
// Explicit specialization of f with 'double' with the  
// template argument deduced:  
//  
template<> void f<double>(double &d)  
{  
d++;
//std::cout << "double" << std::endl;
}  


template<typename T>
void test_VA(T &data) {
   f(data);
   std::cout << data << std::endl;
}

template<typename T, typename... Args>
void test_VA(T &data ,Args & ... args) {
   data++;
   std::cout << data << std::endl;
   test_VA(args...);
}

int main()
{
double t=0.0;
char x = 'c';
int a=10,b=20,c=30,d=40;
//f(t);
//f(x);
//f(a);
//test_VA(a,b,c);
//long sum = adder(1, 2, 3, 8, 7);
test_VA(a,x,t,d);
//	std::cout << a <<" " << b <<" " << c << " " << d << std::endl;
	
}
