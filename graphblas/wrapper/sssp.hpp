#ifndef GRAPHBLAS_WRAPPER_SSSP_H_
#define GRAPHBLAS_WRAPPER_SSSP_H_

#include <vector>

namespace graphblas {
namespace wrapper {

int sssp(int nrows, int ncols, int nvals, int* row_ptr, int* col_ind, float* vals, int source, std::vector<float>& result, int argc, char** argv);

}
}

#endif
