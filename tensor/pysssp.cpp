
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

#include "torch/torch.h"
#include "graphblas/wrapper/sssp.hpp"


struct Args {
  std::vector<char*> args_;

  Args() {
    char* n = (char*)std::malloc(1);
    n[0] = 0;
    args_.push_back(n);
  }
  ~Args() {
    for (int i = 0; i < args_.size(); i ++)
      free(args_[i]);
  }
  void add(const char* arg) {
    args_.push_back(const_cast<char*>(arg));
  }
  int argc() {
    return args_.size();
  }
  char** argv() {
    return args_.data();
  }
};


int main(int argc, char** argv) {
  const char* input_file = argv[argc-1];

  int nrows, ncols;
  std::vector<int> row_indices, col_indices;
  std::vector<float> vals;
  std::string line;
  std::vector<std::string> flds;
  std::ifstream input_fp(input_file);
  int line_cnt = 0;
  while (std::getline(input_fp, line)) {
    boost::split(flds, line, boost::is_any_of(" "));
    if (flds.size() == 0 || flds[0].size() == 0 || flds[0][0] == '%')
      continue;
    if (line_cnt == 0) {
      nrows = std::stoi(flds[0]);
      ncols = std::stoi(flds[1]);
    } else {
      row_indices.push_back(std::stoi(flds[0]));
      col_indices.push_back(std::stoi(flds[1]));
      vals.push_back(std::stof(flds[2]));
    }
    line_cnt ++;
  }
  input_fp.close();
  if (nrows * ncols != vals.size()) {
    std::cout << "Error: nrows * ncols != vals.size()" << std::endl;
    return 1;
  }
  torch::Device device(torch::DeviceType::CUDA, 0);
  /* 
  torch::Tensor data = torch::zeros({nrows, ncols}, torch::TensorOptions(torch::kFloat32).device(device));
  torch::Tensor rowptr = torch::arange(0, nrows * ncols + 1, ncols, torch::TensorOptions(torch::kInt32).device(device));
  torch::Tensor colind = torch::arange(0, ncols, 1, torch::TensorOptions(torch::kInt32).device(device)).repeat(nrows).contiguous();
  for (int i = 0; i < row_indices.size(); i++) {
    data[row_indices[i] - 1][col_indices[i] - 1] = vals[i];
  }
  */
  torch::Tensor data = torch::zeros(20, torch::TensorOptions(torch::kFloat32).device(device));
  torch::Tensor rowptr = torch::arange(0, 21, 4, torch::TensorOptions(torch::kInt32).device(device));
  torch::Tensor colind = torch::zeros(20, torch::TensorOptions(torch::kInt32).device(device));
  int j = 0;
  for (int i = 0; i < row_indices.size(); i++) {
    if (row_indices[i] == col_indices[i])
      continue;
    data[j] = vals[i];
    colind[j] = col_indices[i] - 1;
    j ++;
  }

  std::cout << "Data:" << std::endl;
  std::cout << data << std::endl;
  std::cout << "RowPtr:" << std::endl;
  std::cout << rowptr << std::endl;
  std::cout << "ColInd:" << std::endl;
  std::cout << colind << std::endl;
  std::vector<float> result;
  Args args;
  int status = graphblas::wrapper::sssp(
        nrows, ncols, 20,
        (int*)rowptr.data_ptr(), (int*)colind.data_ptr(), (float*)data.data_ptr(),
        0, result, args.argc(), args.argv());
  std::cout << "status=" << status << std::endl;
  std::cout << "result: ";
  for (int i = 0; i < result.size(); i ++)
    std::cout << "[" << i << "]" << result[i] << ", ";
  std::cout << std::endl;
  return 0;
}
