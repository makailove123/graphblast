#ifndef GRB_USE_CUDA
#define GRB_USE_CUDA
#endif
#define private public

#include <iostream>
#include <vector>

#include <boost/program_options.hpp>
#include "graphblas/graphblas.hpp"
#include "graphblas/algorithm/sssp.hpp"
#include "graphblas/algorithm/common.hpp"

#include "graphblas/wrapper/sssp.hpp"

namespace graphblas {
namespace wrapper {

int sssp(int nrows, int ncols, int nvals, int* row_ptr, int* col_ind, float* vals, int source, std::vector<float>& result, int argc, char** argv)
{
  setEnv("GRB_SPARSE_MATRIX_FORMAT", backend::GrB_SPARSE_MATRIX_CSRONLY);
  po::variables_map vm;
  parseArgs(argc, argv, &vm);
  Descriptor desc;
  CHECK(desc.loadArgs(vm));
  Matrix<float> a(nrows, ncols);
  CHECK(a.build(row_ptr, col_ind, vals, nvals));
  std::cout << "********************************" << std::endl;
  a.print(1);
  std::cout << "********************************" << std::endl;
  Vector<float> v(nrows);
  algorithm::sssp(&v, &a, source, &desc);
  CHECK(v.extractTuples(&result, &nrows));
  return 0;
}

}
}
