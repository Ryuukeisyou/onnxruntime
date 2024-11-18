// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "grid_sample.h"
#include "grid_sample_impl.h"

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, VERSION, LAYOUT, DOMAIN)          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      GridSample,                                                  \
      DOMAIN,                                                      \
      VERSION,                                                     \
      T,                                                           \
      kCudaExecutionProvider,                                      \
                                    (*KernelDefBuilder::Create())                                \
                                        .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
                                        .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      onnxruntime::contrib::cuda::GridSample<T, LAYOUT>);

REGISTER_KERNEL_TYPED(float, 1, LAYOUT_NCHW, kMSDomain)

#ifdef ENABLE_CUDA_NHWC_OPS
REGISTER_KERNEL_TYPED(float, 16, LAYOUT_NHWC, kMSInternalNHWCDomain)
#endif

template <typename T, bool IsNHWC>
GridSample<T, IsNHWC>::GridSample(const OpKernelInfo& info) : CudaKernel(info) {
  std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
  std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
  align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
  ORT_ENFORCE(mode_str == "bilinear" || mode_str == "nearest" || mode_str == "bicubic",
              "mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
  ORT_ENFORCE(padding_mode_str == "zeros" || padding_mode_str == "border" || padding_mode_str == "reflection",
              "padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
  if (mode_str == "bicubic") {
    mode_i_ = 2;
  } else if (mode_str == "nearest") {
    mode_i_ = 1;
  } else {
    mode_i_ = 0;
  }
  if (padding_mode_str == "reflection") {
    padding_mode_i_ = 2;
  } else if (padding_mode_str == "border") {
    padding_mode_i_ = 1;
  } else {
    padding_mode_i_ = 0;
  }
}

template <typename T, bool IsNHWC>
Status GridSample<T, IsNHWC>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const auto& dims_input = X->Shape().GetDims();
  const Tensor* Grid = context->Input<Tensor>(1);
  const auto& dims_grid = Grid->Shape().GetDims();

  const auto& input_dims = X->Shape();
  const auto& grid_dims = Grid->Shape();

  int64_t data_dims = input_dims.NumDimensions() - 2;
  ORT_ENFORCE(static_cast<int64_t>(grid_dims.NumDimensions()) == data_dims + 2,
              "grid dimensions must be ", data_dims + 2, "for input dimension of ", data_dims);

  ORT_ENFORCE(grid_dims[grid_dims.NumDimensions() - 1] == data_dims,
              "Last dimension of grid: ", grid_dims[grid_dims.NumDimensions() - 1], ", expect ", data_dims);

  ORT_ENFORCE(input_dims.NumDimensions() == 4 || input_dims.NumDimensions() == 5, "Only 4-D or 5-D tensor is supported");

  auto N = input_dims[0];
  auto C = input_dims[1];
  ORT_ENFORCE(grid_dims[0] == N, "Grid batch size ", grid_dims[0], " does not match input batch size ", N);

  if (input_dims.NumDimensions() == 5) {
    ORT_ENFORCE(mode_i_ != 3, "Only support GridSample Cubic mode in 4-D cases.");
  }

  if (data_dims == 2) {
    using Ch = Channels<IsNHWC>;

    TensorShapeVector dims_output(4);
    dims_output[Ch::N] = dims_input[Ch::N];
    dims_output[Ch::C] = dims_input[Ch::C];
    dims_output[Ch::H] = dims_grid[1 /* Grid::H */];
    dims_output[Ch::W] = dims_grid[2 /* Grid::W */];
    Tensor* Y = context->Output(0, dims_output);
    // Return early if the output tensor is going to be of size 0
    if (Y->Shape().Size() == 0) {
      return Status::OK();
    }

    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
    GridSampleImpl<CudaT, IsNHWC>(
        Stream(context),
        reinterpret_cast<const CudaT*>(X->Data<T>()),
        reinterpret_cast<const CudaT*>(Grid->Data<T>()),
        mode_i_,
        padding_mode_i_,
        align_corners_,
        dims_input.data(),
        dims_grid[1],
        dims_grid[2],
        Y_data);
  } else if (data_dims == 3) {
    auto D_out = grid_dims[1];
    auto H_out = grid_dims[2];
    auto W_out = grid_dims[3];
    TensorShape Y_shape = {N, C, D_out, H_out, W_out};
    Tensor* Y = context->Output(0, Y_shape);
    // Return early if the output tensor is going to be of size 0
    if (Y->Shape().Size() == 0) {
      return Status::OK();
    }

    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
    GridSampleImpl3D<CudaT, IsNHWC>(
        Stream(context),
        reinterpret_cast<const CudaT*>(X->Data<T>()),
        reinterpret_cast<const CudaT*>(Grid->Data<T>()),
        mode_i_,
        padding_mode_i_,
        align_corners_,
        input_dims.GetDims().data(),
        grid_dims[1],
        grid_dims[2],
        grid_dims[3],
        Y_data);
  }
  return Status::OK();
}
}  // namespace cuda
}  // namespace contrib

namespace cuda {
REGISTER_KERNEL_TYPED(float, 16, LAYOUT_NCHW, kOnnxDomain)
}  // namespace cuda
}  // namespace onnxruntime
