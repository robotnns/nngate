#include "multidim_type.h"

int main ()
{
  double * a = new double[1000];
 /* int ligne 
  int colone*/
  (*(a+2+10))=0.2225;
MultiDimMatrix<double> C;
MultiDimMatrix<double> A(2,2);
MultiDimMatrix<double> B(2,2);
A.set2D(u_long(0),u_long(0),double(10.22));
C = A.add(B);
C.printVec();

  return 0;

}
