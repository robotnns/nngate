#ifndef NNG_TYPE_H
#define NNG_TYPE_H

#include <vector>
#include <cmath>
#include <stdlib.h>
#include "dlib/matrix/matrix.h"


namespace nng
{
    typedef std::vector<double> Vectord;
    typedef dlib::matrix<double,0,1> column_vector;
    const double epsilon = 0.000001;
};

#endif
