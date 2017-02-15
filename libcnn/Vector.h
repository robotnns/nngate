#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <cmath>
#include <stdlib.h>

#include "nng_type.h"
#include "Matrix2d.h"


namespace nng{


    
    
	class Matrix2d;
    class  Vector
    {
	public:
			Vector(size_t len);
            Vector(size_t len, const double& initial);
            Vector(const Vectord& v);
            Vector(const Vector& rhs);
			Vector(Vector&& rhs);
			Vector& operator=(Vector&& rhs);
            ~Vector(){};

            // Access the individual elements
            double& operator()(const size_t& index);
            const double& operator()(const size_t& index) const;
            double& operator[](const size_t& index);
            const double& operator[](const size_t& index) const;

            // Vector/Vector operations
            Vector operator+(const Vector& rhs);
            Vector operator-(const Vector& rhs);
            Vector operator/(const Vector& rhs);
            Vector operator-() const;
            Vector dot(const Vector& v);

            // Vector/scalar operations
            Vector operator+(const double& rhs);
            Vector operator-(const double& rhs);
            Vector operator*(const double& rhs);
            Vector operator/(const double& rhs);

            void print();
            Vector getSegment(size_t start_index, size_t len) const;
			Vector getTail(size_t start_index) const;
            void setSegment(size_t start_index, size_t len, const Vector& v);
            Vector concatenate(const Vectord& v);
            Vector concatenate(const Vector& v);
            Vector kl_divergence(Vector& v);
            double sum();
			Vector power(const double exponent);
			double norm2();
			Vector normalize();
			Matrix2d toDiagonal();

            void setLength(size_t len) {length = len;}
            size_t get_length() const {return length;}
            Vectord& getVector(){return vec;}
            const Vectord& getVector() const {return vec;}

        private:
            Vectord vec;
            size_t length;
    };
    
}

#endif