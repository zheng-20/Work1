#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "knnquerydilate_cuda_kernel.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void knnquerydilate_cuda(int b, int n, int m, int nsample, int select, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    float *dist2 = dist2_tensor.data<float>();

    // cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); // Adapt to torch1.5 and above

    knnquerydilate_cuda_launcher(b, n, m, nsample, select, xyz, new_xyz, idx, dist2, stream);
}
