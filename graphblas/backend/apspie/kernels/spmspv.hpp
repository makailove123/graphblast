#ifndef GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
#define GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP

#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

namespace graphblas
{
namespace backend
{
  // Memory requirements: (4|V|+5|E|)*GrB_THRESHOLD
  //   -GrB_THRESHOLD is defined in graphblas/types.hpp
  //
  //  -> d_cscColBad    |V|*GrB_THRESHOLD
  //  -> d_cscColGood   |V|*GrB_THRESHOLD
  //  -> d_cscColDiff   |V|*GrB_THRESHOLD
  //  -> d_index        |V|*GrB_THRESHOLD
  //  -> d_cscVecInd    |E|*GrB_THRESHOLD (u_ind)
  //  -> d_cscSwapInd   |E|*GrB_THRESHOLD
  //  -> d_cscVecVal    |E|*GrB_THRESHOLD
  //  -> d_cscTempVal   |E|*GrB_THRESHOLD (u_val)
  //  -> d_cscSwapVal   |E|*GrB_THRESHOLD
  //  -> w_ind          |E|*GrB_THRESHOLD
  //  -> w_val          |E|*GrB_THRESHOLD
  //  -> d_temp_storage runtime constant
  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvApspie( Index*            w_ind,
                     W*                w_val,
                     Index*            w_nvals,
                     AccumOp           accum_op,
                     a                 identity,
                     MulOp             mul_op,
                     AddOp             add_op,
                     Index             A_nrows,
                     Index             A_nvals,
                     const Index*      A_csrRowPtr,
                     const Index*      A_csrColInd,
                     const a*          A_csrVal,
                     const Index*      u_ind,
                     const U*          u_val,
                     const Index*      u_nvals,
                     const Descriptor* desc )
  {
    // Get descriptor parameters for nthreads
    /*Desc_value ta_mode, tb_mode, nt_mode;
    CHECK( desc->get(GrB_TA, &ta_mode) );
    CHECK( desc->get(GrB_TB, &tb_mode) );
    CHECK( desc->get(GrB_NT, &nt_mode) );

    const int ta = static_cast<int>(ta_mode);
    const int tb = static_cast<int>(tb_mode);
    const int nt = static_cast<int>(nt_mode);

    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (ta*A_nrows+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

		//Step 1) Gather from CSR graph into one big array  |     |  |
		// 1. Extracts the row lengths we are interested in 3  3  3  2  3  1
		//  -> d_cscColBad  |V|/2
		// 2. Scans them, giving the offset from 0          0  3  6  8
		//  -> d_cscColGood |V|/2
		// 3. Extracts the col indices starts we are interested in 0  6  9
		//  -> d_cscColBad  |V|/2
		// 4. Extracts the neighbour lists
		//  -> d_cscVecInd  |E|/2 (u_ind)
		//  -> d_cscVecVal  |E|/2
		IntervalGather( h_cscVecCount, u_ind, d->d_index, h_cscVecCount, d->d_cscColDiff, d->d_cscColBad, context );
		mgpu::Scan<mgpu::MgpuScanTypeExc>( d->d_cscColBad, h_cscVecCount, 0, mgpu::plus<int>(), (int*)0, &total, d->d_cscColGood, context );
		IntervalGather( h_cscVecCount, u_ind, d->d_index, h_cscVecCount, d_cscColPtr, d->d_cscColBad, context );

		//printf("Processing %d nodes frontier size: %d\n", h_cscVecCount, total);

    //Step 2) Vector Portion
		// a) naive method
		//   -IntervalExpand into frontier-length list
		//      1. Gather the elements indexed by d_cscVecInd
		//      2. Expand the elements to memory set by d_cscColGood
		//   -Element-wise multiplication with frontier
    //  -> d_cscTempVal |E|/2 (u_val)
    //  -> d_cscSwapVal |E|/2
		//IntervalGather( h_cscVecCount, d->d_cscVecInd, d->d_index, h_cscVecCount, d_randVec, d->d_cscTempVal, context );
		IntervalExpand( total, d->d_cscColGood, u_val, h_cscVecCount, d->d_cscSwapVal, context );

		//Step 3) Matrix Structure Portion
    //  -> d_cscVecInd  |E|/2
    //  -> d_cscTempVal |E|/2
		IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscRowInd, d->d_cscVecInd, context );
		IntervalGather( total, d->d_cscColBad, d->d_cscColGood, h_cscVecCount, d_cscVal, d->d_cscTempVal, context );

		//Step 4) Element-wise multiplication
		elementMult<<<NBLOCKS, NTHREADS>>>( total, d->d_cscSwapVal, d->d_cscTempVal, d->d_cscVecVal );

		//Step 1-4) custom kernel method (1 single kernel)
		//  modify spmvCsrBinary() to use Indirect load and stop after expand phase
    //  output: 1) index array 2) value array

		//Step 5) Sort step
    //  -> d_cscSwapInd |E|/2
    //  -> d_cscVecVal  |E|/2
    //  -> d_cscSwapVal |E|/2
		cub::DeviceRadixSort::SortPairs( d->d_temp_storage, temp_storage_bytes, 
        d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, 
        total );
		CUDA( cudaMalloc(&d->d_temp_storage, temp_storage_bytes) );
		cub::DeviceRadixSort::SortPairs( d->d_temp_storage, temp_storage_bytes, 
        d->d_cscVecInd, d->d_cscSwapInd, d->d_cscVecVal, d->d_cscSwapVal, 
        total );
		//MergesortKeys(d_cscVecInd, total, mgpu::less<int>(), context);

		//Step 6) Gather the rand values
		//gather<<<NBLOCKS,NTHREADS>>>( total, d_cscVecVal, d_randVec, d_cscVecVal );

		//Step 7) Segmented Reduce By Key
		ReduceByKey( d->d_cscSwapInd, d->d_cscSwapVal, total, (float)0, 
        mgpu::plus<float>(), mgpu::equal_to<int>(), w_ind, w_val, 
        &h_cscVecCount, (int*)0, context );

		//printf("Current iteration: %d nonzero vector, %d edges\n",  h_cscVecCount, total);

		//Step 8) Reset dense flag array
    //  -> d_mmResult  |V|
		//preprocessFlag<<<NBLOCKS,NTHREADS>>>( d_mmResult, m );

    //Step 9) Sparse Vector to Dense Vector
		//scatterFloat<<<NBLOCKS,NTHREADS>>>( h_cscVecCount, d->d_cscSwapInd, d->d_cscSwapVal, d_mmResult );
    //return total;*/
    return GrB_SUCCESS;
  }

  // Memory requirements: 2|E|*GrB_THRESHOLD
  //   -GrB_THRESHOLD is defined in graphblas/types.hpp
  // 
  //  -> d_cscSwapInd   |E|*GrB_THRESHOLD [2*A_nrows:|E|*GrB_THRESHOLD]
  //  -> d_cscSwapVal   |E|*GrB_THRESHOLD [2*A_nrows+|E|*GrB_THRESHOLD]
  //  -> d_temp_storage runtime constant
  //
  // TODO: can lower 2|E|*GrB_THRESHOLD memory requirement further by doing 
  //       external memory sorting
  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvApspieLB( Index*            w_ind,
                       W*                w_val,
                       Index*            w_nvals,
                       AccumOp           accum_op,
                       a                 identity,
                       MulOp             mul_op,
                       AddOp             add_op,
                       Index             A_nrows,
                       Index             A_nvals,
                       const Index*      A_csrRowPtr,
                       const Index*      A_csrColInd,
                       const a*          A_csrVal,
                       const Index*      u_ind,
                       const U*          u_val,
                       const Index*      u_nvals,
                       const Descriptor* desc )
  {
    // Get descriptor parameters for nthreads
    Desc_value ta_mode, tb_mode, nt_mode;
    CHECK( desc->get(GrB_TA, &ta_mode) );
    CHECK( desc->get(GrB_TB, &tb_mode) );
    CHECK( desc->get(GrB_NT, &nt_mode) );

    const int ta = static_cast<int>(ta_mode);
    const int tb = static_cast<int>(tb_mode);
    const int nt = static_cast<int>(nt_mode);

    dim3 NT, NB;
    NT.x = nt;
    NT.y = 1;
    NT.z = 1;
    NB.x = (ta*A_nrows+nt-1)/nt;
    NB.y = 1;
    NB.z = 1;

		//Step 1) Gather from CSR graph into one big array  |     |  |
    //Step 2) Vector Portion
		//   -IntervalExpand into frontier-length list
		//      1. Gather the elements indexed by d_cscVecInd
		//      2. Expand the elements to memory set by d_cscColGood
		//   -Element-wise multiplication with frontier
		//Step 3) Matrix Structure Portion
		//Step 4) Element-wise multiplication
		//Step 1-4) custom kernel method (1 single kernel)
		//  modify spmvCsrIndirectBinary() to stop after expand phase
    //  output: 1) expanded index array 2) expanded value array
    mgpu::SpmspvCsrIndirectBinary(A_csrVal, A_csrColInd, A_nvals, A_csrRowPtr, 
        A_nrows, u_ind, u_val, *u_nvals, true, w_ind, w_val, w_nvals, (T)0, 
        mul_op, *(desc->d_context_));

		//Step 5) Sort step
    //  -> d_cscSwapInd |E|/2
    //  -> d_cscSwapVal |E|/2
    size_t temp_storage_bytes;
    float  size           = A_nvals*GrB_THRESHOLD+1;
    Descriptor* desc_t    = const_cast<Descriptor*>(desc);
    Index* d_cscSwapInd   = (Index*) desc_t->d_buffer_+2*A_nrows;
    T*     d_cscSwapVal   = (T*)     desc_t->d_buffer_+2*A_nrows+(int) size;
    void*  d_temp_storage = desc_t->d_buffer_+2*A_nrows+2*(int) size;
		
    cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, 
        w_ind, d_cscSwapInd, w_val, d_cscSwapVal, *w_nvals );
		CUDA( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
		cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, 
        w_ind, d_cscSwapInd, w_val, d_cscSwapVal, *w_nvals );
		//MergesortKeys(d_cscVecInd, total, mgpu::less<int>(), desc->d_context_);

		//Step 7) Segmented Reduce By Key
    Index  w_nvals_t      = 0;
		ReduceByKey( d_cscSwapInd, d_cscSwapVal, *w_nvals, (float)0, 
        add_op, mgpu::equal_to<int>(), w_ind, w_val, 
        &w_nvals_t, (int*)0, *(desc->d_context_) );
    *w_nvals = w_nvals_t;

		//printf("Current iteration: %d nonzero vector, %d edges\n",  h_cscVecCount, total);
    return GrB_SUCCESS;
  }

  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvGunrockLB( Index*            w_ind,
                        W*                w_val,
                        Index*            w_nvals,
                        AccumOp           accum_op,
                        a                 identity,
                        MulOp             mul_op,
                        AddOp             add_op,
                        Index             A_nrows,
                        Index             A_nvals,
                        const Index*      A_csrRowPtr,
                        const Index*      A_csrColInd,
                        const a*          A_csrVal,
                        const Index*      u_ind,
                        const U*          u_val,
                        const Index*      u_nvals,
                        const Descriptor* desc )
  {
  }

  template <bool UseScmp, bool UseAccum, bool UseRepl,
            typename W, typename a, typename U,
            typename AccumOp, typename MulOp, typename AddOp>
  Info spmspvGunrockTWC( Index*            w_ind,
                         W*                w_val,
                         Index*            w_nvals,
                         AccumOp           accum_op,
                         a                 identity,
                         MulOp             mul_op,
                         AddOp             add_op,
                         Index             A_nrows,
                         Index             A_nvals,
                         const Index*      A_csrRowPtr,
                         const Index*      A_csrColInd,
                         const a*          A_csrVal,
                         const Index*      u_ind,
                         const U*          u_val,
                         const Index*      u_nvals,
                         const Descriptor* desc )
  {
  }

  //__global__ void filterKernel();

}  // backend
}  // graphblas

#endif  // GRB_BACKEND_APSPIE_KERNELS_SPMSPV_HPP
