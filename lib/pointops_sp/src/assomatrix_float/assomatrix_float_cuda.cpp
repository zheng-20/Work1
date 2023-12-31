#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "assomatrix_float_cuda_kernel.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void assomatrix_float_cuda(int b, int n, int m, int ks, at::Tensor val_c_tensor, at::Tensor idx_c_tensor, at::Tensor cid_tensor, at::Tensor idx_tensor, at::Tensor cnt_tensor) //
{
    CHECK_INPUT(idx_c_tensor);
    CHECK_INPUT(cid_tensor);

    const float *val_c = val_c_tensor.data<float>();
    const int *idx_c = idx_c_tensor.data<int>();
    const int *cid = cid_tensor.data<int>();
    float *idx = idx_tensor.data<float>();
    int *cnt = cnt_tensor.data<int>();

    // cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); // Adapt to torch1.5 and above

    assomatrix_float_cuda_launcher(b, n, m, ks, val_c, idx_c, cid, idx, cnt, stream);
}
