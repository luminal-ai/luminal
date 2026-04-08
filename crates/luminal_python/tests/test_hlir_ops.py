from typing import Callable

import torch
import torch._dynamo
from test_models import (
    AddAddTestModel,
    AddConstantTestModel,
    AddTestModel,
    # And model
    AndTestModel,
    CastBoolToFloatModel,
    # Cast models
    CastDoubleToFloatModel,
    CastInComputationGraphModel,
    CastInt32ToFloatModel,
    CastInt64ToFloatModel,
    CastNegativeValuesModel,
    CastScalarValueModel,
    CastWith2DTensorModel,
    CeilInExpressionModel,
    CeilNegativeModel,
    # Ceil models
    CeilTestModel,
    ClipMinOnlyTestModel,
    # Clip models
    ClipTestModel,
    # Concat models
    ConcatAxis0Model,
    ConcatAxis1Model,
    ConcatInExpressionModel,
    ConcatNegativeAxisModel,
    ConcatThreeTensorsModel,
    Constant1DArrayFloatModel,
    Constant2DMatrixFloatModel,
    ConstantBoolConversionModel,
    ConstantFloat64ConversionModel,
    ConstantInt32ConversionModel,
    ConstantInt64ConversionModel,
    ConstantInt64RawDataModel,
    ConstantMultipleInGraphModel,
    ConstantNegativeValuesModel,
    ConstantRawDataFloatModel,
    # Constant models
    ConstantScalarFloatModel,
    ConstantZeroValueModel,
    CosTestModel,
    DivTestModel,
    # Equal models
    EqualBroadcastModel,
    EqualTestModel,
    EqualWithConstantModel,
    # Erf model
    ErfTestModel,
    # Expand model
    ExpandTestModel,
    FloorInExpressionModel,
    FloorNegativeModel,
    # Floor models
    FloorTestModel,
    # Gather models
    Gather1DModel,
    Gather2DAxis0Model,
    Gather2DAxis1Model,
    GatherConstantFoldModel,
    # GatherElements model
    GatherElementsTestModel,
    GatherElementsLargeTestModel,
    GatherEmbeddingModel,
    GatherNegativeIndicesModel,
    # Gemm model
    GemmTestModel,
    # GreaterOrEqual models
    GreaterOrEqualTestModel,
    GreaterOrEqualWithConstantModel,
    # Greater models
    GreaterTestModel,
    GreaterWithConstantModel,
    # IsNaN model
    IsNaNTestModel,
    # LayerNormalization model
    LayerNormTestModel,
    LessBroadcastModel,
    # LessOrEqual models
    LessOrEqualTestModel,
    LessOrEqualWithConstantModel,
    # Less models
    LessTestModel,
    LessWithConstantModel,
    LinearLayerModel,
    # Multi-op chain models
    ManualLayerNormModel,
    # MatMul models
    MatMul2DModel,
    MatMulBatchedModel,
    # Max models
    MaxTestModel,
    MaxWithConstantModel,
    # Min models
    MinTestModel,
    MinWithConstantModel,
    MLPBlockModel,
    ModByConstantModel,
    # Mod models
    ModTestModel,
    MulTestModel,
    # Not model
    NotTestModel,
    # OneHot model
    OneHotTestModel,
    # Or model
    OrTestModel,
    PowByConstantModel,
    # Pow models
    PowTestModel,
    ReduceMax3DAxis1Model,
    ReduceMaxAllAxesModel,
    # ReduceMax models
    ReduceMaxAxis0Model,
    ReduceMaxAxis1Model,
    ReduceMaxInExpressionModel,
    ReduceMaxKeepDimsModel,
    ReduceMaxMultiAxisKeepDimsModel,
    ReduceMaxMultiAxisModel,
    ReduceMaxNegativeAxisModel,
    ReduceMean3DAxis1Model,
    ReduceMeanAllAxesModel,
    # ReduceMean models
    ReduceMeanAxis0Model,
    ReduceMeanAxis1Model,
    ReduceMeanInExpressionModel,
    ReduceMeanKeepDimsModel,
    ReduceMeanMultiAxisKeepDimsModel,
    ReduceMeanMultiAxisModel,
    ReduceMeanNegativeAxisModel,
    ReduceMin3DAxis1Model,
    ReduceMinAllAxesModel,
    # ReduceMin models
    ReduceMinAxis0Model,
    ReduceMinAxis1Model,
    ReduceMinInExpressionModel,
    ReduceMinKeepDimsModel,
    ReduceMinMultiAxisKeepDimsModel,
    ReduceMinMultiAxisModel,
    ReduceMinNegativeAxisModel,
    ReduceSum3DAxis1Model,
    ReduceSumAllAxesModel,
    # ReduceSum models
    ReduceSumAxis0Model,
    ReduceSumAxis1Model,
    ReduceSumInExpressionModel,
    ReduceSumKeepDimsModel,
    ReduceSumMultiAxisKeepDimsModel,
    ReduceSumMultiAxisModel,
    ReduceSumNegativeAxisModel,
    # Activation function models
    ReluTestModel,
    Reshape3Dto2DModel,
    ReshapeAfterOpsModel,
    ReshapeInExpressionModel,
    ReshapeInferFirstDimModel,
    ReshapeInferLastDimModel,
    ReshapeRoundtripModel,
    ReshapeTo3DModel,
    # Reshape models
    ReshapeToFlatModel,
    ReshapeToMatrixModel,
    ScaledDotProductModel,
    ScatterElementsAxis0TestModel,
    # ScatterElements models
    ScatterElementsTestModel,
    # ScatterND model
    ScatterNDTestModel,
    ShapeReshapeBatchFlattenModel,
    ShapeReshapeKeepBatchModel,
    SigmoidTestModel,
    SinTestModel,
    SliceMultiAxisTestModel,
    # Slice models
    SliceTestModel,
    SoftmaxDim0TestModel,
    # Softmax models
    SoftmaxTestModel,
    # Split model
    SplitTestModel,
    SqrtTestModel,
    SqueezeAllDimsModel,
    # Squeeze models
    SqueezeAxisModel,
    SqueezeInExpressionModel,
    SqueezeMultipleAxesModel,
    SqueezeNegativeAxisModel,
    SubTestModel,
    TanhTestModel,
    TopKIndicesTestModel,
    # TopK models
    TopKValuesTestModel,
    Transpose3DTestModel,
    Transpose4DTestModel,
    TransposeInExpressionModel,
    TransposeReverseTestModel,
    TransposeTestModel,
    TrilDiagonalTestModel,
    # Trilu models
    TrilTestModel,
    TriuDiagonalTestModel,
    TriuTestModel,
    # Unsqueeze models
    UnsqueezeAxis0Model,
    UnsqueezeMiddleModel,
    # Where models
    WhereSelfSelectModel,
    WhereTestModel,
    WhereWithConstantModel,
    # Xor model
    XorTestModel,
)

from luminal import luminal_backend


def test_add(device: torch.device):
    add_test_model: torch.nn.Module = AddTestModel().to(device)
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_linear_layer(device: torch.device):
    add_test_model: torch.nn.Module = LinearLayerModel().to(device)
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_mul(device: torch.device):
    add_test_model: torch.nn.Module = MulTestModel().to(device)
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )

    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_div(device: torch.device):
    div_test_model: torch.nn.Module = DivTestModel().to(device)
    div_test_mode_compiled: Callable = torch.compile(
        div_test_model, backend=luminal_backend
    )

    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = div_test_model(x)
    output: torch.Tensor = div_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_add_add(device: torch.device):
    add_test_model: torch.nn.Module = AddAddTestModel().to(device)
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)
    other_x: torch.Tensor = torch.rand((5), device=device)
    assert torch.allclose(add_test_mode_compiled(other_x), add_test_model(other_x))


def test_add_broadcast(device: torch.device):
    add_test_model: torch.nn.Module = AddTestModel().to(device)
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand(5, device=device)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_add_constant(device: torch.device):
    add_test_model: torch.nn.Module = AddConstantTestModel().to(device)
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand(5, device=device)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_sub(device: torch.device):
    sub_test_model: torch.nn.Module = SubTestModel().to(device)
    sub_test_mode_compiled = torch.compile(sub_test_model, backend=luminal_backend)
    x = torch.rand((10, 10), device=device)
    output = sub_test_mode_compiled(x)
    original = sub_test_model(x)
    assert torch.allclose(output, original)


def test_sub_broadcast(device: torch.device):
    sub_test_model: torch.nn.Module = SubTestModel().to(device)
    sub_test_mode_compiled = torch.compile(sub_test_model, backend=luminal_backend)
    x = torch.rand((10, 10), device=device)
    output = sub_test_mode_compiled(x)
    original = sub_test_model(x)
    assert torch.allclose(output, original)


def test_sqrt(device: torch.device):
    sqrt_test_model: torch.nn.Module = SqrtTestModel().to(device)
    sqrt_test_model_compiled = torch.compile(sqrt_test_model, backend=luminal_backend)
    x = torch.rand(100, device=device)
    output = sqrt_test_model_compiled(x)
    original = sqrt_test_model(x)
    assert torch.allclose(output, original)


def test_sin(device: torch.device):
    sin_test_model: torch.nn.Module = SinTestModel().to(device)
    sin_test_model_compiled = torch.compile(sin_test_model, backend=luminal_backend)
    x = torch.rand(100, device=device)
    output = sin_test_model_compiled(x)
    original = sin_test_model(x)
    assert torch.allclose(output, original)


def test_cos(device: torch.device):
    cos_test_model: torch.nn.Module = CosTestModel().to(device)
    cos_test_model_compiled = torch.compile(cos_test_model, backend=luminal_backend)
    x = torch.rand(100, device=device)
    output = cos_test_model_compiled(x)
    original = cos_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_2d(device: torch.device):
    """Test basic 2D matrix transpose."""
    transpose_test_model: torch.nn.Module = TransposeTestModel().to(device)
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 10), device=device)
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_3d(device: torch.device):
    """Test 3D transpose with dimension permutation."""
    transpose_test_model: torch.nn.Module = Transpose3DTestModel().to(device)
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_4d(device: torch.device):
    """Test 4D transpose (common in conv nets: NCHW -> NHWC)."""
    transpose_test_model: torch.nn.Module = Transpose4DTestModel().to(device)
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((1, 3, 224, 224), device=device)
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_reverse(device: torch.device):
    """Test default transpose (reverse all dimensions)."""
    transpose_test_model: torch.nn.Module = TransposeReverseTestModel().to(device)
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((2, 3, 4, 5), device=device)
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_in_expression(device: torch.device):
    """Test transpose within a larger computational graph."""
    transpose_test_model: torch.nn.Module = TransposeInExpressionModel().to(device)
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 10), device=device)
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_square_matrix(device: torch.device):
    """Test transpose of square matrix (edge case)."""
    transpose_test_model: torch.nn.Module = TransposeTestModel().to(device)
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5), device=device)
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


# ========== ONNX Constant Node Tests ==========
# These tests verify the parse_constant_node function in ops_parse.rs
# which handles ONNX Constant nodes (nodes with embedded data in attributes)


def test_constant_scalar_float(device: torch.device):
    """Test scalar constant (broadcasts to input shape)."""
    model: torch.nn.Module = ConstantScalarFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_1d_array_float(device: torch.device):
    """Test 1D array constant."""
    model: torch.nn.Module = Constant1DArrayFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_2d_matrix_float(device: torch.device):
    """Test 2D matrix constant."""
    model: torch.nn.Module = Constant2DMatrixFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_raw_data_float(device: torch.device):
    """Test raw binary data format (chunks_exact(4) code path)."""
    model: torch.nn.Module = ConstantRawDataFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_int32_conversion(device: torch.device):
    """Test INT32 -> f32 conversion."""
    model: torch.nn.Module = ConstantInt32ConversionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_int64_conversion(device: torch.device):
    """Test INT64 -> f32 conversion."""
    model: torch.nn.Module = ConstantInt64ConversionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([2.0, 3.0, 4.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_float64_conversion(device: torch.device):
    """Test FLOAT64 -> f32 conversion."""
    model: torch.nn.Module = ConstantFloat64ConversionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10.0, 20.0, 30.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_bool_conversion(device: torch.device):
    """Test BOOL -> f32 conversion (0.0/1.0)."""
    model: torch.nn.Module = ConstantBoolConversionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_int64_raw_data(device: torch.device):
    """Test raw binary format for INT64 (chunks_exact(8) code path)."""
    model: torch.nn.Module = ConstantInt64RawDataModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10.0, 20.0, 30.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_negative_values(device: torch.device):
    """Test negative constants."""
    model: torch.nn.Module = ConstantNegativeValuesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([100.0, 200.0, 300.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_zero_value(device: torch.device):
    """Test all-zero constant."""
    model: torch.nn.Module = ConstantZeroValueModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_multiple_in_graph(device: torch.device):
    """Test multiple Constant nodes in one graph."""
    model: torch.nn.Module = ConstantMultipleInGraphModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([5.0, 6.0, 7.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Cast Node Tests ==========
# These tests verify the parse_cast_node function in ops_parse.rs
# which handles ONNX Cast nodes (type conversion operations)


def test_cast_double_to_float(device: torch.device):
    """Test downcast: Double (FLOAT64) -> Float."""
    model: torch.nn.Module = CastDoubleToFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor(
        [1.123456789, 2.987654321, 3.555555555, 4.111111111], dtype=torch.float64
    ).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_int32_to_float(device: torch.device):
    """Test INT32 -> Float conversion."""
    model: torch.nn.Module = CastInt32ToFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_int64_to_float(device: torch.device):
    """Test INT64 -> Float conversion."""
    model: torch.nn.Module = CastInt64ToFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([100, 200, 300, 400], dtype=torch.int64).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_bool_to_float(device: torch.device):
    """Test BOOL -> Float conversion (non-zero -> 1.0, zero -> 0.0)."""
    model: torch.nn.Module = CastBoolToFloatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor(
        [True, False, True, False, True, False], dtype=torch.bool
    ).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_in_computation_graph(device: torch.device):
    """Test Cast node followed by an operation (Cast + Add)."""
    model: torch.nn.Module = CastInComputationGraphModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10, 20, 30], dtype=torch.int32).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_with_2d_tensor(device: torch.device):
    """Test Cast with 2D tensor (matrix)."""
    model: torch.nn.Module = CastWith2DTensorModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_negative_values(device: torch.device):
    """Test Cast with negative integer values."""
    model: torch.nn.Module = CastNegativeValuesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([-10, -5, 0, 5, 10], dtype=torch.int32).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_scalar_value(device: torch.device):
    """Test Cast with scalar (single element)."""
    model: torch.nn.Module = CastScalarValueModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([42.123456], dtype=torch.float64).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Mod Node Tests ==========


def test_mod(device: torch.device):
    """Test basic element-wise modulo."""
    model: torch.nn.Module = ModTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device) * 10.0
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_mod_broadcast(device: torch.device):
    """Test modulo with broadcasting (1D input vs 2D weight)."""
    model: torch.nn.Module = ModTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(5, device=device) * 10.0
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_mod_by_constant(device: torch.device):
    """Test modulo by a constant tensor (exercises Mod + Constant nodes)."""
    model: torch.nn.Module = ModByConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([7.0, 9.0, 11.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Floor Node Tests ==========


def test_floor(device: torch.device):
    """Test element-wise floor on positive floats."""
    model: torch.nn.Module = FloorTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.2, 2.7, 3.0, 4.9, 5.5]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_floor_negative(device: torch.device):
    """Test floor with negative values."""
    model: torch.nn.Module = FloorNegativeModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([-1.2, -2.7, -0.1, 0.9, -3.5]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_floor_in_expression(device: torch.device):
    """Test floor followed by mul (floor as part of a larger graph)."""
    model: torch.nn.Module = FloorInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.5, 2.8, 3.3, 4.1]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Ceil Node Tests ==========


def test_ceil(device: torch.device):
    """Test element-wise ceil on positive floats."""
    model: torch.nn.Module = CeilTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.2, 2.7, 3.0, 4.9, 5.5]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_ceil_negative(device: torch.device):
    """Test ceil with negative values."""
    model: torch.nn.Module = CeilNegativeModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([-1.2, -2.7, -0.1, 0.9, -3.5]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_ceil_in_expression(device: torch.device):
    """Test ceil followed by mul (ceil as part of a larger graph)."""
    model: torch.nn.Module = CeilInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.5, 2.8, 3.3, 4.1]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Reshape Node Tests ==========
# These tests verify parse_reshape_node and parse_shape_node in ops_parse.rs


def test_reshape_2d_to_1d(device: torch.device):
    """Reshape (3, 4) -> (12,) via reshape(-1)."""
    model: torch.nn.Module = ReshapeToFlatModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_1d_to_2d(device: torch.device):
    """Reshape (12,) -> (3, 4) with explicit target shape."""
    model: torch.nn.Module = ReshapeToMatrixModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(12, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_1d_to_3d(device: torch.device):
    """Reshape (24,) -> (2, 3, 4) with explicit 3D target shape."""
    model: torch.nn.Module = ReshapeTo3DModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(24, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_infer_last_dim(device: torch.device):
    """Reshape (12,) -> (3, 4) with -1 inferring last dimension."""
    model: torch.nn.Module = ReshapeInferLastDimModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(12, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_infer_first_dim(device: torch.device):
    """Reshape (12,) -> (3, 4) with -1 inferring first dimension."""
    model: torch.nn.Module = ReshapeInferFirstDimModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(12, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_3d_to_2d(device: torch.device):
    """Reshape (2, 3, 4) -> (2, 12) collapsing last two dims."""
    model: torch.nn.Module = Reshape3Dto2DModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_in_expression(device: torch.device):
    """Reshape (12,) -> (2, 6) then add weight tensor."""
    model: torch.nn.Module = ReshapeInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(12, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_roundtrip(device: torch.device):
    """Reshape (3, 4) -> (12,) -> (3, 4) via two Reshape nodes."""
    model: torch.nn.Module = ReshapeRoundtripModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_reshape_after_ops(device: torch.device):
    """Multiply by 2.0 then reshape (2, 3, 4) -> (24,)."""
    model: torch.nn.Module = ReshapeAfterOpsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_shape_reshape_batch_flatten(device: torch.device):
    """Dynamic batch flatten: x.reshape(x.shape[0], -1) on (2, 3, 4) -> (2, 12)."""
    model: torch.nn.Module = ShapeReshapeBatchFlattenModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_shape_reshape_view_batch(device: torch.device):
    """Dynamic batch flatten via view: x.view(x.size(0), -1) on (2, 3, 4) -> (2, 12)."""
    model: torch.nn.Module = ShapeReshapeKeepBatchModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Less Node Tests ==========
# These tests verify parse_less_node in ops_parse.rs


def test_less(device: torch.device):
    """Test element-wise less-than comparison against a weight tensor."""
    model: torch.nn.Module = LessTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_less_broadcast(device: torch.device):
    """Test less-than with broadcasting (1D input broadcasts against 2D weight)."""
    model: torch.nn.Module = LessBroadcastModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(5, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_less_with_constant(device: torch.device):
    """Test less-than against an inline constant (exercises Less + Constant nodes)."""
    model: torch.nn.Module = LessWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([0.1, 0.5, 0.9]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Equal Node Tests ==========
# These tests verify parse_equal_node in ops_parse/binary.rs


def test_equal(device: torch.device):
    """Test element-wise equality against a stored weight tensor."""
    model: torch.nn.Module = EqualTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.randint(0, 3, (5, 5)).float().to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_equal_broadcast(device: torch.device):
    """Test equality with broadcasting (1D input broadcasts against 2D weight)."""
    model: torch.nn.Module = EqualBroadcastModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.randint(0, 3, (5,)).float().to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_equal_with_constant(device: torch.device):
    """Test equality against an inline constant (exercises Equal + Constant nodes)."""
    model: torch.nn.Module = EqualWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 0.5, 3.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Gather Node Tests ==========
# These tests verify parse_gather_node in ops_parse.rs


def test_gather_1d(device: torch.device):
    """Test Gather on 1D data with constant indices (exercises 1D flat gather path)."""
    model: torch.nn.Module = Gather1DModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(6, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_gather_embedding(device: torch.device):
    """Test Gather via nn.Embedding (axis=0 gather on 2D weight with integer indices)."""
    model: torch.nn.Module = GatherEmbeddingModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([0, 2, 5, 1]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_gather_2d_axis0(device: torch.device):
    """Test Gather on 2D weight along axis=0 with runtime integer indices (row selection)."""
    model: torch.nn.Module = Gather2DAxis0Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([2, 0, 4, 1]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_gather_2d_axis1(device: torch.device):
    """Test Gather on 2D float input along axis=1 (exercises permute-gather-permute strategy)."""
    model: torch.nn.Module = Gather2DAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(4, 5, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_gather_negative_indices(device: torch.device):
    """Test Gather with negative runtime indices (-1 for last element, etc.)."""
    model: torch.nn.Module = GatherNegativeIndicesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([-1, -2, 0, 2]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_gather_constant_fold(device: torch.device):
    """Test Gather constant folding: both data and indices known at graph-build time."""
    model: torch.nn.Module = GatherConstantFoldModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Squeeze Node Tests ==========
# These tests verify parse_squeeze_node in ops_parse.rs


def test_squeeze_axis(device: torch.device):
    """Test squeezing a single size-1 leading dimension."""
    model: torch.nn.Module = SqueezeAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(1, 3, 4, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_squeeze_all_dims(device: torch.device):
    """Test squeezing all size-1 dimensions with no explicit axes."""
    model: torch.nn.Module = SqueezeAllDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(1, 3, 1, 4, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_squeeze_multiple_axes(device: torch.device):
    """Test squeezing multiple explicit size-1 dimensions."""
    model: torch.nn.Module = SqueezeMultipleAxesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(1, 3, 1, 4, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_squeeze_negative_axis(device: torch.device):
    """Test squeezing a trailing size-1 dimension using a negative axis."""
    model: torch.nn.Module = SqueezeNegativeAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(3, 4, 1, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_squeeze_in_expression(device: torch.device):
    """Test squeeze followed by an add (squeeze as part of a larger graph)."""
    model: torch.nn.Module = SqueezeInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(1, 5, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX ReduceSum Node Tests ==========


def test_reduce_sum_axis0(device: torch.device):
    """Test sum reduction along axis 0."""
    model: torch.nn.Module = ReduceSumAxis0Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_axis1(device: torch.device):
    """Test sum reduction along axis 1."""
    model: torch.nn.Module = ReduceSumAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_keepdims(device: torch.device):
    """Test sum reduction along axis 1 with keepdim=True."""
    model: torch.nn.Module = ReduceSumKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_all_axes(device: torch.device):
    """Test sum reduction over all axes (produces a scalar)."""
    model: torch.nn.Module = ReduceSumAllAxesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_3d_axis1(device: torch.device):
    """Test sum reduction along axis 1 for a 3D tensor."""
    model: torch.nn.Module = ReduceSum3DAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_multi_axis(device: torch.device):
    """Test sum reduction along multiple axes (0 and 2)."""
    model: torch.nn.Module = ReduceSumMultiAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_multi_axis_keepdims(device: torch.device):
    """Test sum reduction along axes (0, 2) with keepdim=True."""
    model: torch.nn.Module = ReduceSumMultiAxisKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_negative_axis(device: torch.device):
    """Test sum reduction along axis -1 (last axis)."""
    model: torch.nn.Module = ReduceSumNegativeAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_sum_in_expression(device: torch.device):
    """Test ReduceSum used in a larger expression (mean via sum / n)."""
    model: torch.nn.Module = ReduceSumInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ========== ONNX ReduceMax Node Tests ==========


def test_reduce_max_axis0(device: torch.device):
    """Test max reduction along axis 0."""
    model: torch.nn.Module = ReduceMaxAxis0Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_axis1(device: torch.device):
    """Test max reduction along axis 1."""
    model: torch.nn.Module = ReduceMaxAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_keepdims(device: torch.device):
    """Test max reduction along axis 1 with keepdim=True."""
    model: torch.nn.Module = ReduceMaxKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_all_axes(device: torch.device):
    """Test max reduction over all axes (produces a scalar)."""
    model: torch.nn.Module = ReduceMaxAllAxesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_3d_axis1(device: torch.device):
    """Test max reduction along axis 1 for a 3D tensor."""
    model: torch.nn.Module = ReduceMax3DAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_multi_axis(device: torch.device):
    """Test max reduction along multiple axes (0 and 2)."""
    model: torch.nn.Module = ReduceMaxMultiAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_multi_axis_keepdims(device: torch.device):
    """Test max reduction along axes (0, 2) with keepdim=True."""
    model: torch.nn.Module = ReduceMaxMultiAxisKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_negative_axis(device: torch.device):
    """Test max reduction along axis -1 (last axis)."""
    model: torch.nn.Module = ReduceMaxNegativeAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_max_in_expression(device: torch.device):
    """Test ReduceMax used in a larger expression (max * 2.0)."""
    model: torch.nn.Module = ReduceMaxInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ========== ONNX ReduceMin Node Tests ==========
# These tests verify parse_reduce_min_node in ops_parse/reduction.rs


def test_reduce_min_axis0(device: torch.device):
    """Test min reduction along axis 0."""
    model: torch.nn.Module = ReduceMinAxis0Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_axis1(device: torch.device):
    """Test min reduction along axis 1."""
    model: torch.nn.Module = ReduceMinAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_keepdims(device: torch.device):
    """Test min reduction along axis 1 with keepdim=True."""
    model: torch.nn.Module = ReduceMinKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_all_axes(device: torch.device):
    """Test min reduction over all axes (produces a scalar)."""
    model: torch.nn.Module = ReduceMinAllAxesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_3d_axis1(device: torch.device):
    """Test min reduction along axis 1 for a 3D tensor."""
    model: torch.nn.Module = ReduceMin3DAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_multi_axis(device: torch.device):
    """Test min reduction along multiple axes (0 and 2)."""
    model: torch.nn.Module = ReduceMinMultiAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_multi_axis_keepdims(device: torch.device):
    """Test min reduction along axes (0, 2) with keepdim=True."""
    model: torch.nn.Module = ReduceMinMultiAxisKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_negative_axis(device: torch.device):
    """Test min reduction along axis -1 (last axis)."""
    model: torch.nn.Module = ReduceMinNegativeAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_min_in_expression(device: torch.device):
    """Test ReduceMin used in a larger expression (min * 2.0)."""
    model: torch.nn.Module = ReduceMinInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ========== ONNX ReduceMean Node Tests ==========
# These tests verify parse_reduce_mean_node in ops_parse/reduction.rs


def test_reduce_mean_axis0(device: torch.device):
    """Test mean reduction along axis 0."""
    model: torch.nn.Module = ReduceMeanAxis0Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_axis1(device: torch.device):
    """Test mean reduction along axis 1."""
    model: torch.nn.Module = ReduceMeanAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_keepdims(device: torch.device):
    """Test mean reduction along axis 1 with keepdim=True."""
    model: torch.nn.Module = ReduceMeanKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_all_axes(device: torch.device):
    """Test mean reduction over all axes (produces a scalar)."""
    model: torch.nn.Module = ReduceMeanAllAxesModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_3d_axis1(device: torch.device):
    """Test mean reduction along axis 1 for a 3D tensor."""
    model: torch.nn.Module = ReduceMean3DAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_multi_axis(device: torch.device):
    """Test mean reduction along multiple axes (0 and 2)."""
    model: torch.nn.Module = ReduceMeanMultiAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_multi_axis_keepdims(device: torch.device):
    """Test mean reduction along axes (0, 2) with keepdim=True."""
    model: torch.nn.Module = ReduceMeanMultiAxisKeepDimsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_negative_axis(device: torch.device):
    """Test mean reduction along axis -1 (last axis)."""
    model: torch.nn.Module = ReduceMeanNegativeAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_reduce_mean_in_expression(device: torch.device):
    """Test ReduceMean used in a larger expression (mean * 2.0)."""
    model: torch.nn.Module = ReduceMeanInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ========== ONNX Pow Node Tests ==========
# These tests verify parse_pow_node in ops_parse/binary.rs


def test_pow(device: torch.device):
    """Test basic element-wise power."""
    model: torch.nn.Module = PowTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device) + 0.1
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, rtol=1e-4, atol=1e-4)


def test_pow_broadcast(device: torch.device):
    """Test power with broadcasting (1D input broadcasts against 2D weight)."""
    model: torch.nn.Module = PowTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(5, device=device) + 0.1
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, rtol=1e-4, atol=1e-4)


def test_pow_by_constant(device: torch.device):
    """Test power by an inline constant (exercises Pow + Constant nodes)."""
    model: torch.nn.Module = PowByConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([2.0, 3.0, 4.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, rtol=1e-4, atol=1e-4)


# ========== ONNX Where Node Tests ==========
# These tests verify parse_where_node in ops_parse/binary.rs


def test_where(device: torch.device):
    """Test element-wise where with condition selecting from weight buffers."""
    model: torch.nn.Module = WhereTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = (
        torch.rand((5, 5), device=device) - 0.5
    )  # mix of positive and negative
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_where_self_select(device: torch.device):
    """Test where selecting between input and its negation (abs-like pattern)."""
    model: torch.nn.Module = WhereSelfSelectModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = (
        torch.rand((5, 5), device=device) - 0.5
    )  # mix of positive and negative
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_where_with_constant(device: torch.device):
    """Test where with inline constant tensors as branches (exercises Where + Constant nodes)."""
    model: torch.nn.Module = WhereWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([-0.5, 0.5, -1.0]).to(device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Max Node Tests ==========
# These tests verify parse_max_node in ops_parse/binary.rs


def test_max(device: torch.device):
    """Test element-wise maximum against a stored weight tensor."""
    model: torch.nn.Module = MaxTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_max_with_constant(device: torch.device):
    """Test element-wise maximum against an inline constant (exercises Max + Constant nodes)."""
    model: torch.nn.Module = MaxWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(5, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Min Node Tests ==========
# These tests verify parse_min_node in ops_parse/binary.rs


def test_min(device: torch.device):
    """Test element-wise minimum against a stored weight tensor."""
    model: torch.nn.Module = MinTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_min_with_constant(device: torch.device):
    """Test element-wise minimum against an inline constant (exercises Min + Constant nodes)."""
    model: torch.nn.Module = MinWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(5, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Concat Node Tests ==========
# These tests verify parse_concat_node in ops_parse/movement.rs


def test_concat_axis0(device: torch.device):
    """Test concatenation along axis 0 (stacking rows)."""
    model: torch.nn.Module = ConcatAxis0Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_concat_axis1(device: torch.device):
    """Test concatenation along axis 1 (stacking columns)."""
    model: torch.nn.Module = ConcatAxis1Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_concat_three_tensors(device: torch.device):
    """Test concatenation of three tensors along axis 0."""
    model: torch.nn.Module = ConcatThreeTensorsModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_concat_negative_axis(device: torch.device):
    """Test concatenation with negative axis (-1 = last axis)."""
    model: torch.nn.Module = ConcatNegativeAxisModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_concat_in_expression(device: torch.device):
    """Test concat followed by an element-wise operation."""
    model: torch.nn.Module = ConcatInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ONNX Softmax Node Tests ==========
# These tests verify parse_softmax_node in ops_parse/unary.rs


def test_softmax(device: torch.device):
    """Test softmax along last dimension (default axis=-1)."""
    model: torch.nn.Module = SoftmaxTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 8), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_softmax_dim0(device: torch.device):
    """Test softmax along dim=0."""
    model: torch.nn.Module = SoftmaxDim0TestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 8), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


# ========== ONNX LessOrEqual Node Tests ==========


def test_less_or_equal(device: torch.device):
    """Test element-wise less-than-or-equal comparison against a stored weight."""
    model: torch.nn.Module = LessOrEqualTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_less_or_equal_with_constant(device: torch.device):
    """Test less-than-or-equal against an inline constant tensor."""
    model: torch.nn.Module = LessOrEqualWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(3, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX GreaterOrEqual Node Tests ==========


def test_greater_or_equal(device: torch.device):
    """Test element-wise greater-than-or-equal comparison against a stored weight."""
    model: torch.nn.Module = GreaterOrEqualTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_greater_or_equal_with_constant(device: torch.device):
    """Test greater-than-or-equal against an inline constant tensor."""
    model: torch.nn.Module = GreaterOrEqualWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(3, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Not Node Tests ==========


def test_not(device: torch.device):
    """Test logical NOT on a boolean tensor derived from comparison."""
    model: torch.nn.Module = NotTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX And Node Tests ==========


def test_and(device: torch.device):
    """Test logical AND between two boolean tensors."""
    model: torch.nn.Module = AndTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Or Node Tests ==========


def test_or(device: torch.device):
    """Test logical OR between two boolean tensors."""
    model: torch.nn.Module = OrTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Xor Node Tests ==========


def test_xor(device: torch.device):
    """Test logical XOR between two boolean tensors."""
    model: torch.nn.Module = XorTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Trilu Node Tests ==========


def test_tril(device: torch.device):
    """Test lower triangular masking (tril, diagonal=0)."""
    model: torch.nn.Module = TrilTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_triu(device: torch.device):
    """Test upper triangular masking (triu, diagonal=0)."""
    model: torch.nn.Module = TriuTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_tril_diagonal(device: torch.device):
    """Test tril with a positive diagonal offset."""
    model: torch.nn.Module = TrilDiagonalTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_triu_diagonal(device: torch.device):
    """Test triu with a negative diagonal offset."""
    model: torch.nn.Module = TriuDiagonalTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== Activation Function Tests ==========


def test_relu(device: torch.device):
    """Test relu with mixed positive/negative inputs (negatives clipped to 0)."""
    model: torch.nn.Module = ReluTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_sigmoid(device: torch.device):
    """Test sigmoid on a 2D tensor (outputs in range [0, 1])."""
    model: torch.nn.Module = SigmoidTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 8), device=device) * 4.0 - 2.0
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_tanh(device: torch.device):
    """Test tanh on a 2D tensor (outputs in range [-1, 1])."""
    model: torch.nn.Module = TanhTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 8), device=device) * 4.0 - 2.0
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


# ========== Clip Tests ==========


def test_clip(device: torch.device):
    """Test clamp with both min and max bounds."""
    model: torch.nn.Module = ClipTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device) * 2.0 - 1.0
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_clip_min_only(device: torch.device):
    """Test clamp with only a min bound (no upper limit)."""
    model: torch.nn.Module = ClipMinOnlyTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 5), device=device) * 2.0 - 1.0
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== Unsqueeze Tests ==========


def test_unsqueeze(device: torch.device):
    """Test unsqueeze along axis 0: (N,) -> (1, N)."""
    model: torch.nn.Module = UnsqueezeAxis0Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(6, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_unsqueeze_middle(device: torch.device):
    """Test unsqueeze along axis 1: (B, N) -> (B, 1, N)."""
    model: torch.nn.Module = UnsqueezeMiddleModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== Greater Tests ==========


def test_greater(device: torch.device):
    """Test element-wise greater-than comparison against a stored weight tensor."""
    model: torch.nn.Module = GreaterTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((5, 5), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_greater_with_constant(device: torch.device):
    """Test greater-than against a scalar constant (exercises Greater + Constant nodes)."""
    model: torch.nn.Module = GreaterWithConstantModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(8, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== MatMul Tests ==========


def test_matmul_2d(device: torch.device):
    """Test direct 2D matrix multiplication: (3,4) @ (4,5) -> (3,5)."""
    model: torch.nn.Module = MatMul2DModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_matmul_batched(device: torch.device):
    """Test batched 3D matrix multiplication: (2,3,4) @ (2,4,5) -> (2,3,5)."""
    model: torch.nn.Module = MatMulBatchedModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


# ========== Multi-op Chain Tests ==========


def test_layer_norm(device: torch.device):
    """Test manual layer normalization over the last dimension of a (2, 4) tensor."""
    model: torch.nn.Module = ManualLayerNormModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 4), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_scaled_dot_product_attention(device: torch.device):
    """Test scaled dot-product attention pattern: softmax(Q @ K.T / sqrt(d)) @ V."""
    model: torch.nn.Module = ScaledDotProductModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    q: torch.Tensor = torch.rand((4, 8), device=device)
    original: torch.Tensor = model(q)
    output: torch.Tensor = model_compiled(q)
    assert torch.allclose(output, original, atol=1e-5)


def test_mlp_block(device: torch.device):
    """Test two-layer MLP: Linear(8,16) -> ReLU -> Linear(16,4) on input (2,8)."""
    model: torch.nn.Module = MLPBlockModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 8), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    print(original)
    print(output)
    assert torch.allclose(output, original, atol=1e-5)


# ========== ONNX GatherElements Node Tests ==========


def test_gather_elements(device: torch.device):
    """Tests GatherElements op (torch.gather → ONNX GatherElements)."""
    model: torch.nn.Module = GatherElementsTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 3), device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_gather_elements_large(device: torch.device):
    """Tests GatherElements on 4x8 data with 4x3 indices (different axis sizes)."""
    model: torch.nn.Module = GatherElementsLargeTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((4, 8), device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ONNX Expand Node Tests ==========


def test_expand(device: torch.device):
    """Tests Expand op (tensor.expand → ONNX Expand)."""
    model: torch.nn.Module = ExpandTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((1, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ONNX IsNaN Node Tests ==========


def test_isnan(device: torch.device):
    """Tests IsNaN op — all zeros for normal float inputs."""
    model: torch.nn.Module = IsNaNTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 3), device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ONNX LayerNormalization Node Tests ==========


def test_layernorm(device: torch.device):
    """Tests LayerNormalization op (nn.LayerNorm → ONNX LayerNormalization)."""
    model: torch.nn.Module = LayerNormTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((2, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ========== ONNX Gemm Node Tests ==========


def test_gemm(device: torch.device):
    """Tests Gemm op (nn.Linear → ONNX Gemm)."""
    model: torch.nn.Module = GemmTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((3, 4), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ========== ONNX Erf Node Tests ==========


def test_erf(device: torch.device):
    """Tests erf approximation accuracy (atol=1e-4, A&S 7.1.26 max error < 1.5e-7)."""
    model: torch.nn.Module = ErfTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.linspace(-2.0, 2.0, 16, device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-4)


# ========== ONNX Slice Node Tests ==========


def test_slice_1d(device: torch.device):
    """Tests 1D slice x[1:3]."""
    model: torch.nn.Module = SliceTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(5, device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_slice_2d(device: torch.device):
    """Tests 2D multi-axis slice x[1:3, 0:2]."""
    model: torch.nn.Module = SliceMultiAxisTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(4, 4, device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ONNX Split Node Tests ==========


def test_split(device: torch.device):
    """Tests Split with equal-size chunks along axis 1."""
    model: torch.nn.Module = SplitTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(3, 4, device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ONNX TopK Node Tests ==========


def test_topk_values(device: torch.device):
    """Tests TopK values output for 2D tensor along axis=1."""
    model: torch.nn.Module = TopKValuesTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(4, 8, device=device)
    assert torch.allclose(model_compiled(x), model(x))


def test_topk_indices(device: torch.device):
    """Tests TopK indices output for 2D tensor along axis=1."""
    model: torch.nn.Module = TopKIndicesTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand(4, 8, device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ONNX OneHot Node Tests ==========


def test_onehot(device: torch.device):
    """Tests OneHot encoding with depth=5."""
    model: torch.nn.Module = OneHotTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([0, 2, 4, 1, 3], device=device)
    assert torch.allclose(model_compiled(x), model(x))


# ========== ScatterElements Tests ==========


def test_scatter_elements(device: torch.device):
    """Tests scatter along axis=1 using torch.scatter."""
    model: torch.nn.Module = ScatterElementsTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_scatter_elements_axis0(device: torch.device):
    """Tests scatter along axis=0 using torch.scatter."""
    model: torch.nn.Module = ScatterElementsAxis0TestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ScatterND Tests ==========


def test_scatter_nd(device: torch.device):
    """Tests ScatterND via index_put: x[indices] = updates."""
    model: torch.nn.Module = ScatterNDTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=device
    )
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)
