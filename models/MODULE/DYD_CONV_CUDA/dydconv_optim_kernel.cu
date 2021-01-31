#include <torch/types.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

static __host__ __device__ __forceinline__ int floor_div(int a, int b) {
  int c = a / b;

  if (c * b > a) {
    c--;
  }

  return c;
}

enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

struct DWConv2dKernelParams {
  int batch;
  int in_h;
  int in_w;
  int in_channel;

  int kernel_h;
  int kernel_w;

  int up_x;
  int up_y;
  int down_x;
  int down_y;

  int pad_x0;
  int pad_x1;
  int pad_y0;
  int pad_y1;

  int out_h;
  int out_w;
  int out_channel;

  int loop_major;
  int n_out;
};

template <typename scalar_t, DepthwiseConv2dDirection direction, int up_x,
          int up_y, int down_x, int down_y, int kernel_h, int kernel_w,
          int tile_out_h, int tile_out_w>
__global__ void dydconv2d_kernel(scalar_t *out, const scalar_t *input,
                                const scalar_t *kernel,
                                const DWConv2dKernelParams p) {
  const int tile_in_h = ((tile_out_h - 1) * down_y + kernel_h - 1) / up_y + 1;
  const int tile_in_w = ((tile_out_w - 1) * down_x + kernel_w - 1) / up_x + 1;

  //printf("tile_in_h %d tile_out_h %d tile_in_w %d tile_out_w %d\n",tile_in_h, tile_out_h, tile_in_w, tile_out_w);
  __shared__ scalar_t sk[kernel_h][kernel_w];
  __shared__ scalar_t sx[tile_in_h][tile_in_w];

  int minor_idx = blockIdx.x;
  int tile_out_y = minor_idx;
  minor_idx -= tile_out_y;
  tile_out_y *= tile_out_h;
  int tile_out_x_base = blockIdx.y * tile_out_w;
  int major_idx_base = blockIdx.z * p.loop_major;

  const int major_dim = p.batch * p.in_channel;

  if (tile_out_x_base >= p.out_w | tile_out_y >= p.out_h |
      major_idx_base >= major_dim) {
    return;
  }

  for (int loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major & major_idx < major_dim;
       loop_major++, major_idx++) {
    int channel_idx = major_idx % p.in_channel;
       
    for (int tap_idx = threadIdx.x; tap_idx < kernel_h * kernel_w;
         tap_idx += blockDim.x) {
      int ky = tap_idx / kernel_w;
      int kx = tap_idx - ky * kernel_w;
      scalar_t v = 0.0;

      if (kx < p.kernel_w & ky < p.kernel_h) {
        if (direction == DIRECTION_FORWARD) {
          //((major_idx * p.in_h + in_y) * p.in_w + in_x) + minor_idx
          v = kernel[major_idx * p.kernel_w * p.kernel_h + ky * p.kernel_w +
                     kx];
    
        } else {
          v = kernel[major_idx * p.kernel_w * p.kernel_h +
                     (p.kernel_h - 1 - ky) * p.kernel_w +
                     (p.kernel_w - 1 - kx)];
        }
      }

      sk[ky][kx] = v;
    }

    __syncthreads();

    for (int loop_x = 0, tile_out_x = tile_out_x_base;
         loop_x < 1 & tile_out_x < p.out_w;
         loop_x++, tile_out_x += tile_out_w) {
      int tile_mid_x = tile_out_x * down_x + up_x - 1 - p.pad_x0;
      int tile_mid_y = tile_out_y * down_y + up_y - 1 - p.pad_y0;
      int tile_in_x = floor_div(tile_mid_x, up_x);
      int tile_in_y = floor_div(tile_mid_y, up_y);

      for (int in_idx = threadIdx.x; in_idx < tile_in_h * tile_in_w;
           in_idx += blockDim.x) {
        int rel_in_y = in_idx / tile_in_w;
        int rel_in_x = in_idx - rel_in_y * tile_in_w;
        int in_x = rel_in_x + tile_in_x;
        int in_y = rel_in_y + tile_in_y;

        scalar_t v = 0.0;

        if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
          v = input[((major_idx * p.in_h + in_y) * p.in_w + in_x) + minor_idx];
        }

        sx[rel_in_y][rel_in_x] = v;
      }

      __syncthreads();

      for (int out_idx = threadIdx.x; out_idx < tile_out_h * tile_out_w;
           out_idx += blockDim.x) {
        int rel_out_y = out_idx / tile_out_w;
        int rel_out_x = out_idx - rel_out_y * tile_out_w;
        int out_x = rel_out_x + tile_out_x;
        int out_y = rel_out_y + tile_out_y;

        int mid_x = tile_mid_x + rel_out_x * down_x;
        int mid_y = tile_mid_y + rel_out_y * down_y;
        int in_x = floor_div(mid_x, up_x);
        int in_y = floor_div(mid_y, up_y);
        int rel_in_x = in_x - tile_in_x;
        int rel_in_y = in_y - tile_in_y;
        int kernel_x = (in_x + 1) * up_x - mid_x - 1;
        int kernel_y = (in_y + 1) * up_y - mid_y - 1;

        scalar_t v = 0.0;

#pragma unroll
        for (int y = 0; y < kernel_h / up_y; y++)
#pragma unroll
          for (int x = 0; x < kernel_w / up_x; x++) {
            v += sx[rel_in_y + y][rel_in_x + x] *
                 sk[kernel_y + y * up_y][kernel_x + x * up_x];
          }

        if (out_x < p.out_w & out_y < p.out_h) {
          out[((major_idx * p.out_h + out_y) * p.out_w + out_x) + minor_idx] =
              v;
        }
      }
    }
  }
}

DWConv2dKernelParams make_conv2d_params(const torch::Tensor &input,
                                        const torch::Tensor &kernel, int up_h,
                                        int up_w, int down_h, int down_w,
                                        int pad_h0, int pad_h1, int pad_w0,
                                        int pad_w1) {
  DWConv2dKernelParams p;

  p.batch = input.size(0);
  p.in_channel = input.size(1);
  p.in_h = input.size(2);
  p.in_w = input.size(3);
  p.kernel_h = kernel.size(2);
  p.kernel_w = kernel.size(3);
  p.up_x = up_w;
  p.up_y = up_h;
  p.down_x = down_w;
  p.down_y = down_h;
  p.pad_x0 = pad_w0;
  p.pad_x1 = pad_w1;
  p.pad_y0 = pad_h0;
  p.pad_y1 = pad_h1;

  p.out_h = (p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - p.kernel_h + p.down_y) /
            p.down_y;
  p.out_w = (p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - p.kernel_w + p.down_x) /
            p.down_x;
  p.out_channel = p.in_channel;
  p.n_out = p.batch * p.in_channel * p.out_h * p.out_w;

  return p;
}


template <typename scalar_t, DepthwiseConv2dDirection direction, int up_x,
          int up_y, int down_x, int down_y, int kernel_h, int kernel_w,
          int tile_out_h, int tile_out_w>
torch::Tensor dydconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, DWConv2dKernelParams p) {
  int cur_device = -1;
  cudaGetDevice(&cur_device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);
  auto out =
      at::empty({p.batch, p.in_channel, p.out_h, p.out_w}, input.options());

  dim3 block_size;
  dim3 grid_size;

  int major_dim = p.batch * p.in_channel;

  if (tile_out_h > 0 && tile_out_w > 0) {
    p.loop_major = (major_dim - 1) / 16384 + 1;
    block_size = dim3(32 * 8, 1, 1);
    grid_size =
        dim3(((p.out_h - 1) / tile_out_h + 1), (p.out_w - 1) / tile_out_w + 1,
             (major_dim - 1) / p.loop_major + 1);
  }

  dydconv2d_kernel<scalar_t, direction, up_x, up_y, down_x, down_y, kernel_h,
                  kernel_w, tile_out_h, tile_out_w>
      <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                          input.data_ptr<scalar_t>(),
                                          kernel.data_ptr<scalar_t>(), p);
  return out;
}

template <typename scalar_t, DepthwiseConv2dDirection direction>
torch::Tensor dydconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, DWConv2dKernelParams p) {
  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1) {
    if (p.kernel_h <= 3 && p.kernel_w <= 3) {
      return dydconv2d_op<scalar_t, direction, 1, 1, 1, 1, 3, 3, 16, 64>(
          input, kernel, p);

    } else if (p.kernel_h <= 5 && p.kernel_w <= 5) {
      return dydconv2d_op<scalar_t, direction, 1, 1, 1, 1, 5, 5, 16, 64>(
          input, kernel, p);
    } else if (p.kernel_h <= 7 && p.kernel_w <= 7) {
      return dydconv2d_op<scalar_t, direction, 1, 1, 1, 1, 7, 7, 16, 64>(
          input, kernel, p);
    }
  } else if (p.up_x == 2 && p.up_y == 2) {
    if (p.kernel_h <= 4 && p.kernel_w <= 4) {
      return dydconv2d_op<scalar_t, direction, 2, 2, 1, 1, 4, 4, 16, 64>(
          input, kernel, p);
    } else if (p.kernel_h <= 6 && p.kernel_w <= 6) {
      return dydconv2d_op<scalar_t, direction, 2, 2, 1, 1, 6, 6, 16, 64>(
          input, kernel, p);
    } else if (p.kernel_h <= 8 && p.kernel_w <= 8) {
      return dydconv2d_op<scalar_t, direction, 2, 2, 1, 1, 8, 8, 16, 64>(
          input, kernel, p);
    }
  } else if (p.down_x == 2 && p.down_y == 2) {
    if (p.kernel_h <= 4 && p.kernel_w <= 4) {
      return dydconv2d_op<scalar_t, direction, 1, 1, 2, 2, 4, 4, 8, 32>(
          input, kernel, p);
    } else if (p.kernel_h <= 6 && p.kernel_w <= 6) {
      return dydconv2d_op<scalar_t, direction, 1, 1, 2, 2, 6, 6, 8, 32>(
          input, kernel, p);
    } else if (p.kernel_h <= 8 && p.kernel_w <= 8) {
      return dydconv2d_op<scalar_t, direction, 1, 1, 2, 2, 8, 8, 8, 32>(
          input, kernel, p);
    }
  }
}

torch::Tensor dydconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, int up_h, int up_w,
                          int down_h, int down_w, int pad_h0, int pad_h1,
                          int pad_w0, int pad_w1, bool forward) {
  DWConv2dKernelParams p =
      make_conv2d_params(input, kernel, up_h, up_w, down_h, down_w, pad_h0,
                         pad_h1, pad_w0, pad_w1);

  auto x = input.contiguous();
  auto k = kernel.contiguous();

  torch::Tensor out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "dydconv2d", [&] {
    if (forward) {
      out = dydconv2d_op<scalar_t, DIRECTION_FORWARD>(x, k, p);
    } else {
      out = dydconv2d_op<scalar_t, DIRECTION_BACKWARD>(x, k, p);
    }
  });

  return out;
}



