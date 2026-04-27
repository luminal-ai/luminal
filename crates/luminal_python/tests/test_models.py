"""Test models for HLIR ops testing."""

import torch


class SelfAddModel(torch.nn.Module):
    """Adds input to itself (x + x). Preserves input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class AddTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return self.weight + x


class MulTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return self.weight * x


class DivTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return self.weight / x


class AddAddTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight_1", torch.rand((5, 5)))
        self.register_buffer("weight_2", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return self.weight_1 + x + self.weight_2


class AddConstantTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x + 10


class LinearLayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))
        self.register_buffer("bias", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return (self.weight @ x) + self.bias


class SqrtTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sqrt()


class SinTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class CosTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


class SubTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((10, 10)))

    def forward(self, x: torch.Tensor):
        return self.weight - x


class TransposeTestModel(torch.nn.Module):
    """Test basic 2D transpose (matrix transpose)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.t()  # 2D transpose


class Transpose3DTestModel(torch.nn.Module):
    """Test 3D transpose with explicit permutation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(2, 0, 1)  # Rotate dimensions


class Transpose4DTestModel(torch.nn.Module):
    """Test 4D transpose (NCHW -> NHWC)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)  # Common in CNNs


class TransposeReverseTestModel(torch.nn.Module):
    """Test reverse permutation (default transpose behavior)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(x.ndim))
        return x.permute(*reversed(dims))


class TransposeInExpressionModel(torch.nn.Module):
    """Test transpose as part of larger expression."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((10, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (5, 10), transpose to (10, 5), add weight
        return x.t() + self.weight


# ========== Constant Node Test Models ==========
# These models test PT2 Constant node handling via inline tensor literals


class ConstantScalarFloatModel(torch.nn.Module):
    """Test scalar constant (broadcasts to input shape)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor(10.5).to(x.device)
        return x + constant


class Constant1DArrayFloatModel(torch.nn.Module):
    """Test 1D array constant."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(x.device)
        return x * constant


class Constant2DMatrixFloatModel(torch.nn.Module):
    """Test 2D matrix constant."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(x.device)
        return x + constant


class ConstantRawDataFloatModel(torch.nn.Module):
    """Test constant with specific values (tests raw data format)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([7.5, 8.5, 9.5]).to(x.device)
        return x + constant


class ConstantInt32ConversionModel(torch.nn.Module):
    """Test INT32 constant values (PyTorch exports as integers)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).to(x.device)
        return x + constant.float()


class ConstantInt64ConversionModel(torch.nn.Module):
    """Test INT64 constant values."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([100, 200, 300], dtype=torch.int64).to(x.device)
        return x * constant.float()


class ConstantFloat64ConversionModel(torch.nn.Module):
    """Test FLOAT64 (double) constant values."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64).to(x.device)
        return x * constant.float()


class ConstantBoolConversionModel(torch.nn.Module):
    """Test boolean constant values (converted to 0.0/1.0)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([True, False, True, False, True], dtype=torch.bool).to(
            x.device
        )
        return x * constant.float()


class ConstantInt64RawDataModel(torch.nn.Module):
    """Test INT64 constant with large values (tests raw data path)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1000, 2000, 3000], dtype=torch.int64).to(x.device)
        return x + constant.float()


class ConstantNegativeValuesModel(torch.nn.Module):
    """Test negative constant values."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([-5.0, -10.0, -15.0]).to(x.device)
        return x + constant


class ConstantZeroValueModel(torch.nn.Module):
    """Test all-zero constant."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(x.device)
        return x * constant


class ConstantMultipleInGraphModel(torch.nn.Module):
    """Test multiple constants in one graph."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        const1 = torch.tensor([10.0, 20.0, 30.0]).to(x.device)
        const2 = torch.tensor([1.0, 2.0, 3.0]).to(x.device)
        return x + const1 + const2


# ========== Cast Node Test Models ==========
# These models test PT2 Cast node handling via .to(dtype) method


class CastDoubleToFloatModel(torch.nn.Module):
    """Test downcast: Double (FLOAT64) -> Float."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input will be float64, cast to float32
        return x.to(torch.float32)


class CastInt32ToFloatModel(torch.nn.Module):
    """Test INT32 -> Float conversion."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastInt64ToFloatModel(torch.nn.Module):
    """Test INT64 -> Float conversion."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastBoolToFloatModel(torch.nn.Module):
    """Test BOOL -> Float conversion (non-zero -> 1.0, zero -> 0.0)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastInComputationGraphModel(torch.nn.Module):
    """Test Cast node followed by an operation (Cast + Add)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        casted = x.to(torch.float32)
        constant = torch.tensor([2.0, 2.0, 2.0]).to(x.device)
        return casted + constant


class CastWith2DTensorModel(torch.nn.Module):
    """Test Cast with 2D tensor (matrix)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastNegativeValuesModel(torch.nn.Module):
    """Test Cast with negative integer values."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastScalarValueModel(torch.nn.Module):
    """Test Cast with scalar (single element)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


# ========== Mod Node Test Models ==========


class ModTestModel(torch.nn.Module):
    """Tests element-wise modulo with a weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "weight", torch.rand((5, 5)) + 1.0
        )  # ensure non-zero divisor

    def forward(self, x: torch.Tensor):
        return x.fmod(self.weight)


class ModByConstantModel(torch.nn.Module):
    """Tests modulo with an inline constant tensor (PT2 Constant node)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([3.0, 4.0, 5.0]).to(x.device)
        return x.fmod(constant)


# ========== Floor Node Test Models ==========


class FloorTestModel(torch.nn.Module):
    """Tests element-wise floor operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


class FloorNegativeModel(torch.nn.Module):
    """Tests floor with negative values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


class FloorInExpressionModel(torch.nn.Module):
    """Tests floor as part of a larger expression (floor + mul)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x) * 2.0


# ========== Ceil Node Test Models ==========


class CeilTestModel(torch.nn.Module):
    """Tests element-wise ceil operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ceil(x)


class CeilNegativeModel(torch.nn.Module):
    """Tests ceil with negative values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ceil(x)


class CeilInExpressionModel(torch.nn.Module):
    """Tests ceil as part of a larger expression (ceil + mul)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ceil(x) * 2.0


# ========== Reshape Node Test Models ==========
# These models test PT2 Reshape node handling in ops_parse.rs


class ReshapeToFlatModel(torch.nn.Module):
    """Reshape 2D tensor to 1D (full flatten)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1)  # (3, 4) -> (12,)


class ReshapeToMatrixModel(torch.nn.Module):
    """Reshape 1D tensor to 2D matrix."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(3, 4)  # (12,) -> (3, 4)


class ReshapeTo3DModel(torch.nn.Module):
    """Reshape 1D tensor to 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(2, 3, 4)  # (24,) -> (2, 3, 4)


class ReshapeInferLastDimModel(torch.nn.Module):
    """Reshape with -1 to infer last dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(3, -1)  # (12,) -> (3, 4)


class ReshapeInferFirstDimModel(torch.nn.Module):
    """Reshape with -1 to infer first dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, 4)  # (12,) -> (3, 4)


class Reshape3Dto2DModel(torch.nn.Module):
    """Reshape 3D tensor to 2D (common in networks before linear layer)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(2, -1)  # (2, 3, 4) -> (2, 12)


class ReshapeInExpressionModel(torch.nn.Module):
    """Reshape followed by element-wise addition (Reshape in graph)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((2, 6)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = x.reshape(2, 6)  # (12,) -> (2, 6)
        return reshaped + self.weight


class ReshapeRoundtripModel(torch.nn.Module):
    """Reshape to different shape and back (two Reshape nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)  # (3, 4) -> (12,)
        return flat.reshape(3, 4)  # (12,) -> (3, 4)


class ReshapeAfterOpsModel(torch.nn.Module):
    """Apply operation then reshape (op -> Reshape)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = x * 2.0
        return doubled.reshape(-1)  # (2, 3, 4) -> (24,)


class ShapeReshapeBatchFlattenModel(torch.nn.Module):
    """Batch flatten using dynamic shape: preserves first dim, flattens rest."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)  # (2, 3, 4) -> (2, 12)


class ShapeReshapeKeepBatchModel(torch.nn.Module):
    """Use shape of input to reshape, keeping batch dimension dynamic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)  # (2, 3, 4) -> (2, 12)


# ========== Less Node Test Models ==========
# These models test PT2 Less node handling in ops_parse.rs


class LessTestModel(torch.nn.Module):
    """Tests element-wise less-than comparison against a stored weight."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x < self.weight).to(torch.float32)


class LessBroadcastModel(torch.nn.Module):
    """Tests less-than with broadcasting (1D input vs 2D weight)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x < self.weight).to(torch.float32)


class LessWithConstantModel(torch.nn.Module):
    """Tests less-than against an inline constant (PT2 Constant + Less nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.25, 0.5, 0.75]).to(x.device)
        return (x < constant).to(torch.float32)


# ========== Gather Node Test Models ==========
# These models test PT2 Gather node handling in ops_parse.rs


class Gather1DModel(torch.nn.Module):
    """Tests Gather on 1D float data with constant integer indices (1D path)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor([2, 0, 4, 1])
        return x[indices]


class GatherEmbeddingModel(torch.nn.Module):
    """Tests Gather via nn.Embedding (axis=0 gather on 2D weight, integer input)."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding: torch.nn.Embedding = torch.nn.Embedding(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class Gather2DAxis0Model(torch.nn.Module):
    """Tests Gather on 2D weight along axis=0 with runtime integer indices."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand(6, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class Gather2DAxis1Model(torch.nn.Module):
    """Tests Gather on 2D float input along axis=1 (exercises permute-gather-permute strategy)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor([1, 3, 0])
        return x[:, indices]


class GatherNegativeIndicesModel(torch.nn.Module):
    """Tests Gather with negative runtime indices (-1 for last element, etc.)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand(6, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class GatherConstantFoldModel(torch.nn.Module):
    """Tests Gather constant folding: both data and indices are PT2 Constant nodes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).to(x.device)
        indices = torch.tensor([4, 1, 3])
        return x + data[indices]


# ========== Squeeze Node Test Models ==========
# These models test PT2 Squeeze node handling in ops_parse.rs


class SqueezeAxisModel(torch.nn.Module):
    """Tests squeezing a single size-1 leading dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0)  # (1, 3, 4) -> (3, 4)


class SqueezeAllDimsModel(torch.nn.Module):
    """Tests squeezing all size-1 dimensions (no explicit axes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()  # (1, 3, 1, 4) -> (3, 4)


class SqueezeMultipleAxesModel(torch.nn.Module):
    """Tests squeezing multiple explicit size-1 dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0).squeeze(1)  # (1, 3, 1, 4) -> (3, 4)


class SqueezeNegativeAxisModel(torch.nn.Module):
    """Tests squeezing a size-1 dimension using a negative axis index."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)  # (3, 4, 1) -> (3, 4)


class SqueezeInExpressionModel(torch.nn.Module):
    """Tests squeeze as part of a larger expression (squeeze then add)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.squeeze(0)  # (1, 5) -> (5,)
        return squeezed + self.weight


# ========== ReduceSum Node Test Models ==========


class ReduceSumAxis0Model(torch.nn.Module):
    """Tests sum reduction along axis 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=0)  # (4, 5) -> (5,)


class ReduceSumAxis1Model(torch.nn.Module):
    """Tests sum reduction along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)  # (4, 5) -> (4,)


class ReduceSumKeepDimsModel(torch.nn.Module):
    """Tests sum reduction along axis 1 with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)  # (4, 5) -> (4, 1)


class ReduceSumAllAxesModel(torch.nn.Module):
    """Tests sum reduction over all axes (scalar result)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()  # (3, 4) -> scalar


class ReduceSum3DAxis1Model(torch.nn.Module):
    """Tests sum reduction along axis 1 for 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)  # (2, 3, 4) -> (2, 4)


class ReduceSumMultiAxisModel(torch.nn.Module):
    """Tests sum reduction along multiple axes (0 and 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(0, 2))  # (2, 3, 4) -> (3,)


class ReduceSumMultiAxisKeepDimsModel(torch.nn.Module):
    """Tests sum reduction along axes (0, 2) with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(0, 2), keepdim=True)  # (2, 3, 4) -> (1, 3, 1)


class ReduceSumNegativeAxisModel(torch.nn.Module):
    """Tests sum reduction along axis -1 (last axis)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1)  # (3, 4) -> (3,)


class ReduceSumInExpressionModel(torch.nn.Module):
    """Tests ReduceSum used in a larger expression (mean via sum/n)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True) / x.shape[1]  # (3, 4) -> (3, 1) mean


# ========== ReduceMax Node Test Models ==========


class ReduceMaxAxis0Model(torch.nn.Module):
    """Tests max reduction along axis 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=0)  # (4, 5) -> (5,)


class ReduceMaxAxis1Model(torch.nn.Module):
    """Tests max reduction along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1)  # (4, 5) -> (4,)


class ReduceMaxKeepDimsModel(torch.nn.Module):
    """Tests max reduction along axis 1 with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1, keepdim=True)  # (4, 5) -> (4, 1)


class ReduceMaxAllAxesModel(torch.nn.Module):
    """Tests max reduction over all axes (scalar result)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max()  # (3, 4) -> scalar


class ReduceMax3DAxis1Model(torch.nn.Module):
    """Tests max reduction along axis 1 for 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1)  # (2, 3, 4) -> (2, 4)


class ReduceMaxMultiAxisModel(torch.nn.Module):
    """Tests max reduction along multiple axes (0 and 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(0, 2))  # (2, 3, 4) -> (3,)


class ReduceMaxMultiAxisKeepDimsModel(torch.nn.Module):
    """Tests max reduction along axes (0, 2) with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(0, 2), keepdim=True)  # (2, 3, 4) -> (1, 3, 1)


class ReduceMaxNegativeAxisModel(torch.nn.Module):
    """Tests max reduction along axis -1 (last axis)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=-1)  # (3, 4) -> (3,)


class ReduceMaxInExpressionModel(torch.nn.Module):
    """Tests ReduceMax used in a larger expression (max * 2.0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1, keepdim=True) * 2.0  # (3, 4) -> (3, 1)


# ========== ReduceMin Node Test Models ==========


class ReduceMinAxis0Model(torch.nn.Module):
    """Tests min reduction along axis 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=0)  # (4, 5) -> (5,)


class ReduceMinAxis1Model(torch.nn.Module):
    """Tests min reduction along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=1)  # (4, 5) -> (4,)


class ReduceMinKeepDimsModel(torch.nn.Module):
    """Tests min reduction along axis 1 with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=1, keepdim=True)  # (4, 5) -> (4, 1)


class ReduceMinAllAxesModel(torch.nn.Module):
    """Tests min reduction over all axes (scalar result)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.min()  # (3, 4) -> scalar


class ReduceMin3DAxis1Model(torch.nn.Module):
    """Tests min reduction along axis 1 for 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=1)  # (2, 3, 4) -> (2, 4)


class ReduceMinMultiAxisModel(torch.nn.Module):
    """Tests min reduction along multiple axes (0 and 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=(0, 2))  # (2, 3, 4) -> (3,)


class ReduceMinMultiAxisKeepDimsModel(torch.nn.Module):
    """Tests min reduction along axes (0, 2) with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=(0, 2), keepdim=True)  # (2, 3, 4) -> (1, 3, 1)


class ReduceMinNegativeAxisModel(torch.nn.Module):
    """Tests min reduction along axis -1 (last axis)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=-1)  # (3, 4) -> (3,)


class ReduceMinInExpressionModel(torch.nn.Module):
    """Tests ReduceMin used in a larger expression (min * 2.0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=1, keepdim=True) * 2.0  # (3, 4) -> (3, 1)


# ========== ReduceMean Node Test Models ==========


class ReduceMeanAxis0Model(torch.nn.Module):
    """Tests mean reduction along axis 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=0)  # (4, 5) -> (5,)


class ReduceMeanAxis1Model(torch.nn.Module):
    """Tests mean reduction along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)  # (4, 5) -> (4,)


class ReduceMeanKeepDimsModel(torch.nn.Module):
    """Tests mean reduction along axis 1 with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1, keepdim=True)  # (4, 5) -> (4, 1)


class ReduceMeanAllAxesModel(torch.nn.Module):
    """Tests mean reduction over all axes (scalar result)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()  # (3, 4) -> scalar


class ReduceMean3DAxis1Model(torch.nn.Module):
    """Tests mean reduction along axis 1 for 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)  # (2, 3, 4) -> (2, 4)


class ReduceMeanMultiAxisModel(torch.nn.Module):
    """Tests mean reduction along multiple axes (0 and 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=(0, 2))  # (2, 3, 4) -> (3,)


class ReduceMeanMultiAxisKeepDimsModel(torch.nn.Module):
    """Tests mean reduction along axes (0, 2) with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=(0, 2), keepdim=True)  # (2, 3, 4) -> (1, 3, 1)


class ReduceMeanNegativeAxisModel(torch.nn.Module):
    """Tests mean reduction along axis -1 (last axis)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=-1)  # (3, 4) -> (3,)


class ReduceMeanInExpressionModel(torch.nn.Module):
    """Tests ReduceMean used in a larger expression (mean * 2.0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1, keepdim=True) * 2.0  # (3, 4) -> (3, 1)


# ========== Sigmoid Node Test Models ==========


class SigmoidTestModel(torch.nn.Module):
    """Tests sigmoid on a 2D tensor (values in range [0,1] after sigmoid)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class SigmoidInExpressionModel(torch.nn.Module):
    """Tests sigmoid followed by an element-wise multiply."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * 2.0


# ========== Tanh Node Test Models ==========


class TanhTestModel(torch.nn.Module):
    """Tests tanh on a 2D tensor (outputs in [-1,1])."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class TanhInExpressionModel(torch.nn.Module):
    """Tests tanh followed by an element-wise add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) + 1.0


# ========== Relu Node Test Models ==========


class ReluTestModel(torch.nn.Module):
    """Tests relu with mixed positive/negative inputs (negatives clipped to 0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class ReluAllNegativeModel(torch.nn.Module):
    """Tests relu with all-negative inputs (entire output should be zeros)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class ReluInExpressionModel(torch.nn.Module):
    """Tests relu followed by an element-wise multiply."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) * 2.0


# ========== Abs Node Test Models ==========


class AbsTestModel(torch.nn.Module):
    """Tests abs with mixed positive/negative inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class AbsAllNegativeModel(torch.nn.Module):
    """Tests abs with all-negative inputs (output equals negation of input)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class AbsInExpressionModel(torch.nn.Module):
    """Tests abs followed by an element-wise add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) + 1.0


# ========== Neg Node Test Models ==========


class NegTestModel(torch.nn.Module):
    """Tests neg with mixed positive/negative inputs (sign is flipped)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x)


class NegAllPositiveModel(torch.nn.Module):
    """Tests neg with all-positive inputs (output should be all negative)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x)


class NegInExpressionModel(torch.nn.Module):
    """Tests neg followed by an element-wise add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x) + 1.0


# ========== Equal Node Test Models ==========


class EqualTestModel(torch.nn.Module):
    """Tests element-wise equality against a stored weight with discrete values."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randint(0, 3, (5, 5)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.weight).to(torch.float32)


class EqualBroadcastModel(torch.nn.Module):
    """Tests equality with broadcasting (1D input broadcasts against 2D weight)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randint(0, 3, (5, 5)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.weight).to(torch.float32)


class EqualWithConstantModel(torch.nn.Module):
    """Tests equality against an inline constant (exercises Equal + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.0, 2.0, 3.0]).to(x.device)
        return (x == constant).to(torch.float32)


# ========== Clip Node Test Models ==========


class ClipTestModel(torch.nn.Module):
    """Tests clip (clamp) with both min and max bounds."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=-0.5, max=0.5)


class ClipMinOnlyTestModel(torch.nn.Module):
    """Tests clip with only a min bound."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0)


class ClipMaxOnlyTestModel(torch.nn.Module):
    """Tests clip with only a max bound."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, max=0.5)


# ========== Pow Node Test Models ==========


class PowTestModel(torch.nn.Module):
    """Tests element-wise power against a stored weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)) + 1.0)  # positive exponents

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**self.weight


class PowByConstantModel(torch.nn.Module):
    """Tests power by an inline constant tensor (exercises Pow + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([2.0, 3.0, 0.5]).to(x.device)
        return x**constant


# ========== Where Node Test Models ==========


# ========== Max Node Test Models ==========


class MaxTestModel(torch.nn.Module):
    """Tests element-wise maximum against a stored weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, self.weight)


class MaxWithConstantModel(torch.nn.Module):
    """Tests element-wise maximum against an inline constant (PT2 Max + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]).to(x.device)
        return torch.maximum(x, constant)


# ========== Min Node Test Models ==========


class MinTestModel(torch.nn.Module):
    """Tests element-wise minimum against a stored weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x, self.weight)


class MinWithConstantModel(torch.nn.Module):
    """Tests element-wise minimum against an inline constant (PT2 Min + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]).to(x.device)
        return torch.minimum(x, constant)


class WhereTestModel(torch.nn.Module):
    """Tests element-wise where with condition derived from input comparison, selecting from weight buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("true_vals", torch.rand((5, 5)) + 1.0)
        self.register_buffer("false_vals", torch.rand((5, 5)) - 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.0, self.true_vals, self.false_vals)


class WhereSelfSelectModel(torch.nn.Module):
    """Tests where selecting between input x and its negation (abs-like pattern)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.0, x, -x)


class WhereWithConstantModel(torch.nn.Module):
    """Tests where with inline constant tensors for both branches (exercises Where + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.tensor([1.0, 2.0, 3.0]).to(x.device)
        neg = torch.tensor([-1.0, -2.0, -3.0]).to(x.device)
        return torch.where(x > 0.0, pos, neg)


# ========== Concat Node Test Models ==========


class ConcatAxis0Model(torch.nn.Module):
    """Tests concatenation along axis 0 (stacking rows)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=0)


class ConcatAxis1Model(torch.nn.Module):
    """Tests concatenation along axis 1 (stacking columns)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=1)


class ConcatThreeTensorsModel(torch.nn.Module):
    """Tests concatenation of three tensors along axis 0."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight1", torch.rand((2, 4)))
        self.register_buffer("weight2", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight1, self.weight2, x], dim=0)


class ConcatNegativeAxisModel(torch.nn.Module):
    """Tests concatenation with negative axis (-1 = last axis)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=-1)


class ConcatInExpressionModel(torch.nn.Module):
    """Tests concat followed by an element-wise operation."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=0) * 2.0


# ========== Softmax Node Test Models ==========


class SoftmaxTestModel(torch.nn.Module):
    """Tests softmax along the last dimension (default axis=-1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)


class SoftmaxDim0TestModel(torch.nn.Module):
    """Tests softmax along dim=0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=0)


# ========== LessOrEqual Node Test Models ==========


class LessOrEqualTestModel(torch.nn.Module):
    """Tests element-wise less-than-or-equal comparison against a stored weight."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x <= self.weight).to(torch.float32)


class LessOrEqualWithConstantModel(torch.nn.Module):
    """Tests less-than-or-equal against an inline constant (PT2 Constant + LessOrEqual nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.25, 0.5, 0.75]).to(x.device)
        return (x <= constant).to(torch.float32)


# ========== GreaterOrEqual Node Test Models ==========


class GreaterOrEqualTestModel(torch.nn.Module):
    """Tests element-wise greater-than-or-equal comparison against a stored weight."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x >= self.weight).to(torch.float32)


class GreaterOrEqualWithConstantModel(torch.nn.Module):
    """Tests greater-than-or-equal against an inline constant (PT2 Constant + GreaterOrEqual nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.25, 0.5, 0.75]).to(x.device)
        return (x >= constant).to(torch.float32)


# ========== Not Node Test Models ==========


class NotTestModel(torch.nn.Module):
    """Tests logical NOT on a boolean tensor derived from comparison."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(x > 0.5).float()


# ========== And Node Test Models ==========


class AndTestModel(torch.nn.Module):
    """Tests logical AND between two boolean tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(x > 0.3, self.weight > 0.3).float()


# ========== Or Node Test Models ==========


class OrTestModel(torch.nn.Module):
    """Tests logical OR between two boolean tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_or(x > 0.7, self.weight > 0.7).float()


# ========== Xor Node Test Models ==========


class XorTestModel(torch.nn.Module):
    """Tests logical XOR between two boolean tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_xor(x > 0.5, self.weight > 0.5).float()


# ========== Trilu Node Test Models ==========


class TrilTestModel(torch.nn.Module):
    """Tests lower triangular selection (tril, diagonal=0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tril(x)


class TriuTestModel(torch.nn.Module):
    """Tests upper triangular selection (triu, diagonal=0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.triu(x)


class TrilDiagonalTestModel(torch.nn.Module):
    """Tests tril with a positive diagonal offset."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tril(x, diagonal=1)


class TriuDiagonalTestModel(torch.nn.Module):
    """Tests triu with a negative diagonal offset."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.triu(x, diagonal=-1)


# ========== Unsqueeze Node Test Models ==========


class UnsqueezeAxis0Model(torch.nn.Module):
    """Tests unsqueeze along axis 0: adds a leading size-1 dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0)  # (N,) -> (1, N)


class UnsqueezeMiddleModel(torch.nn.Module):
    """Tests unsqueeze along axis 1: inserts a size-1 dimension in the middle."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)  # (B, N) -> (B, 1, N)


# ========== Greater Node Test Models ==========


class GreaterTestModel(torch.nn.Module):
    """Tests element-wise greater-than comparison between two input tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.weight).to(torch.float32)


class GreaterWithConstantModel(torch.nn.Module):
    """Tests greater-than against a scalar constant (PT2 Greater + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.5).to(torch.float32)


# ========== MatMul Test Models ==========


class MatMul2DModel(torch.nn.Module):
    """Tests direct 2D matrix multiplication: (3,4) @ (4,5) -> (3,5)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((4, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


class MatMulBatchedModel(torch.nn.Module):
    """Tests batched 3D matrix multiplication: (2,3,4) @ (2,4,5) -> (2,3,5)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((2, 4, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


# ========== Multi-op Chain Test Models ==========


class ManualLayerNormModel(torch.nn.Module):
    """Tests manual layer normalization: ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)


class ScaledDotProductModel(torch.nn.Module):
    """Tests scaled dot-product attention: softmax(Q @ K.T / sqrt(d)) @ V."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("k", torch.rand((4, 8)))
        self.register_buffer("v", torch.rand((4, 8)))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        d = q.shape[-1] ** 0.5
        scores = torch.matmul(q, self.k.transpose(-2, -1)) / d
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, self.v)


class MLPBlockModel(torch.nn.Module):
    """Tests two-layer MLP using matmul+bias: (8->16, ReLU) -> (16->4)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("w1", torch.rand((8, 16)))
        self.register_buffer("b1", torch.rand(16))
        self.register_buffer("w2", torch.rand((16, 4)))
        self.register_buffer("b2", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.matmul(x, self.w1) + self.b1)
        return torch.matmul(h, self.w2) + self.b2


# ========== GatherElements Node Test Models ==========


class GatherElementsTestModel(torch.nn.Module):
    """Tests element-wise gather along axis=1 using torch.gather (→ PT2 GatherElements)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor([[0, 1, 1], [1, 0, 0]], device=x.device)
        return torch.gather(x, 1, idx)


class GatherElementsLargeTestModel(torch.nn.Module):
    """Tests GatherElements on 4x8 data with 4x3 indices along axis=1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor(
            [[2, 7, 0], [5, 6, 3], [4, 0, 5], [1, 5, 6]], device=x.device
        )
        return torch.gather(x, 1, idx)


# ========== Expand Node Test Models ==========


class ExpandTestModel(torch.nn.Module):
    """Tests broadcasting a (1, 4) tensor to (3, 4) via .expand() (→ PT2 Expand)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(3, 4)


# ========== IsNaN Node Test Models ==========


class IsNaNTestModel(torch.nn.Module):
    """Tests IsNaN — for normal float inputs should return all zeros."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.isnan(x).float()


# ========== LayerNormalization Node Test Models ==========


class LayerNormTestModel(torch.nn.Module):
    """Tests nn.LayerNorm which exports as PT2 LayerNormalization."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ========== Gemm Node Test Models ==========


class GemmTestModel(torch.nn.Module):
    """Tests Gemm: nn.Linear exports as PT2 Gemm (weight transposed)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ========== Erf Node Test Models ==========


class ErfTestModel(torch.nn.Module):
    """Tests erf (Gaussian error function) via tanh approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


# ========== Slice Node Test Models ==========


class SliceTestModel(torch.nn.Module):
    """Tests PT2 Slice: slice axis 0 from index 1 to 3."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[1:3]


class SliceMultiAxisTestModel(torch.nn.Module):
    """Tests PT2 Slice along multiple axes: x[1:3, 0:2]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[1:3, 0:2]


# ========== Split Node Test Models ==========


class SplitTestModel(torch.nn.Module):
    """Tests Split into equal-size chunks along axis 1; combines outputs with addition."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.split(x, 2, dim=1)
        return a + b


# ========== Argsort / MoE Routing Test Models ==========


class ArgsortStableDuplicatesModel(torch.nn.Module):
    """Tests deterministic duplicate ordering for exported argsort."""

    SORT_DIM = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argsort(x, dim=self.SORT_DIM)


class TinyMoERoutingModel(torch.nn.Module):
    """Minimal deterministic MoE-style routing proof for PT2/native and CUDA."""

    TOP_K = 2
    ROUTING_DIM = -1
    ZERO_FILL = 0.0
    DISPATCH_ON = 1
    GROUP_SIZE = 2

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "expert_scale",
            torch.tensor([1.5, -0.5, 2.0, 0.25], dtype=torch.float32),
        )

    def forward(
        self, scores: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        topk_values, topk_indices = torch.topk(scores, self.TOP_K, dim=self.ROUTING_DIM)
        regroup_order = torch.argsort(topk_indices, dim=self.ROUTING_DIM)
        routed_indices = torch.gather(topk_indices, self.ROUTING_DIM, regroup_order)
        routed_values = torch.gather(topk_values, self.ROUTING_DIM, regroup_order)

        expert_scale = self.expert_scale.unsqueeze(0).expand(scores.shape[0], -1)
        gathered_scale = torch.gather(expert_scale, self.ROUTING_DIM, routed_indices)
        weighted = routed_values * gathered_scale

        inactive_mask = torch.bitwise_not(weighted > 0)
        masked_values = weighted.masked_fill(inactive_mask, self.ZERO_FILL)

        slots = torch.zeros_like(routed_indices).scatter(
            self.ROUTING_DIM, regroup_order, self.DISPATCH_ON
        )
        active_slots = torch.bitwise_not(inactive_mask).to(slots.dtype)
        dispatch = slots * active_slots
        group_ids = torch.floor_divide(routed_indices, self.GROUP_SIZE)
        routing_sign = torch.sign(masked_values)
        return (
            routed_indices,
            masked_values,
            dispatch,
            inactive_mask,
            group_ids,
            routing_sign,
        )


# ========== TopK Node Test Models ==========


class TopKValuesTestModel(torch.nn.Module):
    """Tests TopK values output (largest=True) along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _ = torch.topk(x, 3, dim=1)
        return values


class TopKIndicesTestModel(torch.nn.Module):
    """Tests TopK indices output (largest=True) along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(x, 3, dim=1)
        return indices.float()


# ========== OneHot Node Test Models ==========


class OneHotTestModel(torch.nn.Module):
    """Tests OneHot encoding: integer indices -> one-hot matrix with depth=5."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x.long(), num_classes=5).float()


# ========== ScatterElements Node Test Models ==========


class ScatterElementsTestModel(torch.nn.Module):
    """Tests scatter along axis=1 using torch.scatter."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor([[0, 2], [1, 0]], device=x.device)
        src = torch.tensor([[10.0, 30.0], [20.0, 40.0]], device=x.device)
        return x.scatter(1, idx, src)


class ScatterElementsAxis0TestModel(torch.nn.Module):
    """Tests scatter along axis=0 using torch.scatter."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor([[1], [0]], device=x.device)
        src = torch.tensor([[99.0], [88.0]], device=x.device)
        return x.scatter(0, idx, src)


# ========== ScatterND Node Test Models ==========


class ScatterNDTestModel(torch.nn.Module):
    """Tests ScatterND via index_put: x[indices] = updates."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor([0, 2], device=x.device)
        updates = torch.tensor(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], device=x.device
        )
        result = x.clone()
        result[indices] = updates
        return result


# ========== Llama3 Component Test Models ==========


class RMSNormModel(torch.nn.Module):
    """Tests RMS normalization: x * rsqrt(mean(x^2) + eps) * weight.

    PT2 ops: Pow, ReduceMean, Add, Sqrt, Reciprocal, Mul.
    Input: (1, 4, 32) -> Output: (1, 4, 32).
    """

    def __init__(self, hidden_size: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class RotaryEmbeddingModel(torch.nn.Module):
    """Tests rotary position embeddings (RoPE) using rotate-half approach.

    Precomputes cos/sin caches as buffers; at runtime: slice, split halves, rotate.
    PT2 ops: Slice, Unsqueeze, Mul, Sub, Add, Concat.
    Input: (1, 4, 4, 8) [batch, seq, heads, head_dim] -> Output: same shape.
    """

    def __init__(self, head_dim: int = 8, max_seq_len: int = 16) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(2))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLUMLPModel(torch.nn.Module):
    """Tests SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).

    silu(x) = x * sigmoid(x), decomposes to Sigmoid+Mul in PT2.
    Input: (1, 4, 32) -> Output: (1, 4, 32).
    """

    def __init__(self, hidden_size: int = 32, intermediate_size: int = 64) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class CausalSelfAttentionModel(torch.nn.Module):
    """Tests multi-head causal self-attention with additive mask.

    Q/K/V projections -> reshape to heads -> scaled dot-product with causal mask -> output proj.
    Uses additive mask: scores + triu(ones, diag=1) * -1e9.
    Input: (1, 4, 32) -> Output: (1, 4, 32).
    """

    def __init__(
        self, hidden_size: int = 32, num_heads: int = 4, head_dim: int = 8
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.scale = head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = (
            self.q_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = (
            torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) * -1e9
        )
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class LlamaTransformerBlockModel(torch.nn.Module):
    """Tests a full Llama-style transformer block.

    RMSNorm -> RoPE + CausalAttn -> Residual -> RMSNorm -> SwiGLU MLP -> Residual.
    Input: (1, 4, 32) -> Output: (1, 4, 32).
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_heads: int = 4,
        head_dim: int = 8,
        intermediate_size: int = 64,
        eps: float = 1e-6,
        max_seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.input_norm = RMSNormModel(hidden_size, eps)
        self.attn = CausalSelfAttentionModel(hidden_size, num_heads, head_dim)
        self.post_attn_norm = RMSNormModel(hidden_size, eps)
        self.mlp = SwiGLUMLPModel(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.input_norm(x))
        out = h + self.mlp(self.post_attn_norm(h))
        return out


# ---------------------------------------------------------------------------
# Convolution models
# ---------------------------------------------------------------------------


class Conv1dNoPadModel(torch.nn.Module):
    """Conv1d with no padding: output length shrinks by (kernel-1)."""

    KERNEL_SIZE = 3
    PADDING = 0

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            8, 16, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dSamePadModel(torch.nn.Module):
    """Conv1d with same-size padding (output length == input length)."""

    KERNEL_SIZE = 3
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            8, 16, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBiasModel(torch.nn.Module):
    """Conv1d with bias."""

    KERNEL_SIZE = 3
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            8, 16, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dNoPadModel(torch.nn.Module):
    """Conv2d with no padding: output spatial dims shrink by (kernel-1)."""

    KERNEL_SIZE = 3
    PADDING = 0

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 16, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dSamePadModel(torch.nn.Module):
    """Conv2d with same-size padding."""

    KERNEL_SIZE = 3
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 16, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dBiasModel(torch.nn.Module):
    """Conv2d with bias."""

    KERNEL_SIZE = 3
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 16, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dStrideModel(torch.nn.Module):
    """Conv2d with stride=2 (output dims halved)."""

    KERNEL_SIZE = 3
    STRIDE = 2
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3,
            16,
            kernel_size=self.KERNEL_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dDilationModel(torch.nn.Module):
    """Conv2d with dilation=2 and padding chosen to preserve spatial size."""

    KERNEL_SIZE = 3
    DILATION = 2
    PADDING = 2

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            8,
            16,
            kernel_size=self.KERNEL_SIZE,
            dilation=self.DILATION,
            padding=self.PADDING,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv3dSamePadModel(torch.nn.Module):
    """Conv3d with padding=1 to preserve spatial dimensions."""

    KERNEL_SIZE = 3
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(
            4, 8, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthwiseConv1dModel(torch.nn.Module):
    """Depthwise Conv1d as used in Mamba (groups == in_channels)."""

    KERNEL_SIZE = 4
    GROUPS = 16
    PADDING = 3

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            16,
            16,
            kernel_size=self.KERNEL_SIZE,
            groups=self.GROUPS,
            padding=self.PADDING,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal truncation: keep only the first L positions
        return self.conv(x)[:, :, : x.shape[2]]


class DepthwiseConv2dModel(torch.nn.Module):
    """Depthwise Conv2d (groups == in_channels)."""

    KERNEL_SIZE = 3
    GROUPS = 8
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            8,
            8,
            kernel_size=self.KERNEL_SIZE,
            groups=self.GROUPS,
            padding=self.PADDING,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthwiseMultiplierConv2dModel(torch.nn.Module):
    """Depthwise Conv2d with channel multiplier 2 (out_channels = 2 * in_channels)."""

    KERNEL_SIZE = 3
    GROUPS = 8
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            8,
            16,
            kernel_size=self.KERNEL_SIZE,
            groups=self.GROUPS,
            padding=self.PADDING,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GroupedConv2dModel(torch.nn.Module):
    """Conv2d with groups=4 (not depthwise, but grouped)."""

    KERNEL_SIZE = 3
    GROUPS = 4
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            16,
            32,
            kernel_size=self.KERNEL_SIZE,
            groups=self.GROUPS,
            padding=self.PADDING,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GroupedConv2dGroups3Model(torch.nn.Module):
    """Conv2d with groups=3 and ch_per_group=4."""

    KERNEL_SIZE = 3
    GROUPS = 3
    PADDING = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            12,
            12,
            kernel_size=self.KERNEL_SIZE,
            groups=self.GROUPS,
            padding=self.PADDING,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MambaConvBlockModel(torch.nn.Module):
    """Minimal Mamba-style SSM block: Linear -> split -> depthwise Conv1d -> SiLU gate -> Linear.

    This is the core conv pattern used in Mamba / Mamba-2 models.
    """

    def __init__(self, d_model: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        d_inner = d_model * expand
        groups = d_inner
        padding = d_conv - 1
        self.in_proj = torch.nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = torch.nn.Conv1d(
            d_inner,
            d_inner,
            d_conv,
            groups=groups,
            padding=padding,
            bias=True,
        )
        self.out_proj = torch.nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_part = self.conv1d(x_part.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        return self.out_proj(
            torch.nn.functional.silu(x_part) * torch.nn.functional.silu(z)
        )
