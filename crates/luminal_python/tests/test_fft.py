"""Regression coverage for composed PT2 Fourier transform lowerings."""

import torch
from luminal.pt2 import compile as luminal_compile


def _compile(module: torch.nn.Module, inputs, dynamic_shapes=None):
    return luminal_compile(
        module,
        inputs,
        search_iterations=1,
        dynamic_shapes={} if dynamic_shapes is None else dynamic_shapes,
    )


def _outputs(value):
    return value if isinstance(value, tuple) else (value,)


def _assert_compiled_matches(
    module: torch.nn.Module,
    inputs,
    *,
    rtol=6e-4,
    atol=7e-5,
):
    expected = _outputs(module(*inputs))
    actual = _compile(module, inputs)(*inputs)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        torch.testing.assert_close(
            result,
            reference,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


class RealOneDimensionalFfts(torch.nn.Module):
    def forward(self, value):
        return (
            torch.fft.fft(value, n=6, norm="backward"),
            torch.fft.ifft(value, n=4, norm="ortho"),
            torch.fft.rfft(value, norm="forward"),
            torch.fft.ihfft(value, n=6, norm="ortho"),
        )


def test_real_one_dimensional_fft_family():
    value = torch.tensor([1.0, -2.0, 0.5, 3.0, -0.25])
    _assert_compiled_matches(RealOneDimensionalFfts(), (value,))


class ComplexOneDimensionalFfts(torch.nn.Module):
    def forward(self, value):
        return (
            torch.fft.fft(value, norm="ortho"),
            torch.fft.ifft(value, norm="forward"),
            torch.fft.irfft(value, n=6),
            torch.fft.hfft(value, n=6),
            torch.fft.hfft(value, n=7, norm="ortho"),
        )


def test_complex_one_dimensional_fft_family_even_and_odd_lengths():
    value = torch.tensor([1 + 2j, -2 + 0.5j, 0.25 - 1j, 3 + 0j], dtype=torch.complex64)
    _assert_compiled_matches(ComplexOneDimensionalFfts(), (value,))


class HermitianEndpointBehavior(torch.nn.Module):
    def forward(self, value):
        return torch.fft.irfft(value, n=6, norm="forward")


def test_c2r_ignores_dc_and_nyquist_imaginary_components():
    value = torch.tensor(
        [1 + 50j, -2 + 0.5j, 0.25 - 1j, 3 - 80j], dtype=torch.complex64
    )
    _assert_compiled_matches(HermitianEndpointBehavior(), (value,))


class MultiDimensionalFfts(torch.nn.Module):
    def forward(self, value):
        return (
            torch.fft.fftn(value, dim=(0, 2), norm="ortho"),
            torch.fft.rfftn(value, dim=(0, 2), norm="forward"),
            torch.fft.fft2(value, s=(3, 5), dim=(1, 2)),
            torch.fft.ihfftn(value, s=(2, 6), dim=(0, 2)),
        )


def test_multidimensional_fft_family_and_nonfinal_axes():
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) - 7
    _assert_compiled_matches(MultiDimensionalFfts(), (value,), rtol=8e-4, atol=9e-5)


class MultiDimensionalInverseFfts(torch.nn.Module):
    def forward(self, value):
        return (
            torch.fft.ifftn(value, dim=(0, 2), norm="forward"),
            torch.fft.irfftn(value, s=(2, 6), dim=(0, 2), norm="ortho"),
            torch.fft.hfftn(value, s=(2, 6), dim=(0, 2)),
        )


def test_multidimensional_complex_inverse_and_hermitian_ffts():
    real = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 5
    imag = torch.flip(real, dims=(0, 2)) / 7
    value = torch.complex(real, imag)
    _assert_compiled_matches(
        MultiDimensionalInverseFfts(), (value,), rtol=1e-3, atol=1e-4
    )


class ShiftFrequencies(torch.nn.Module):
    def forward(self, value):
        return (
            torch.fft.fftshift(value, dim=(0, 1)),
            torch.fft.ifftshift(value, dim=(0, 1)),
        )


def test_fft_shifts_for_integer_and_complex_values():
    for value in (
        torch.arange(12, dtype=torch.int64).reshape(3, 4),
        torch.complex(
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.arange(12, dtype=torch.float32).reshape(3, 4).flip(1),
        ),
    ):
        _assert_compiled_matches(ShiftFrequencies(), (value,), rtol=0, atol=0)


class DynamicLengthFfts(torch.nn.Module):
    def forward(self, value):
        return (
            torch.fft.fft(value, norm="ortho"),
            torch.fft.rfft(value),
        )


def test_fft_accepts_symbolic_compile_time_length_expression():
    module = DynamicLengthFfts()
    example = torch.arange(4, dtype=torch.float32) - 1
    length = torch.export.Dim("fft_length", min=2, max=8)
    compiled = _compile(module, (example,), dynamic_shapes=({0: length},))

    for size in (4, 6):
        value = torch.linspace(-2, 3, size)
        expected = module(value)
        actual = compiled(value)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(
                result, reference, rtol=6e-4, atol=7e-5, equal_nan=True
            )


class DoublePrecisionFfts(torch.nn.Module):
    def forward(self, real, complex_value):
        return (
            torch.fft.rfft(real, norm="ortho"),
            torch.fft.fft(complex_value),
            torch.fft.irfft(complex_value, n=6),
        )


def test_double_precision_fft_accuracy_and_dtype():
    real = torch.tensor([1.0, -2.0, 0.5, 3.0, -0.25], dtype=torch.float64)
    complex_value = torch.complex(real[:4], torch.flip(real[:4], dims=(0,)))
    _assert_compiled_matches(
        DoublePrecisionFfts(),
        (real, complex_value),
        rtol=1e-7,
        atol=1e-8,
    )


class ShortTimeFourierTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(6))

    def forward(self, value):
        return torch.stft(
            value,
            n_fft=8,
            hop_length=3,
            win_length=6,
            window=self.window,
            center=True,
            normalized=True,
            onesided=True,
            return_complex=True,
        )


def test_stft_decomposition_uses_composed_r2c_lowering():
    value = torch.linspace(-1, 1, 20)
    _assert_compiled_matches(
        ShortTimeFourierTransform(), (value,), rtol=8e-4, atol=9e-5
    )


class InverseShortTimeFourierTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(6))

    def forward(self, value):
        return torch.istft(
            value,
            n_fft=8,
            hop_length=3,
            win_length=6,
            window=self.window,
            center=True,
            normalized=True,
            onesided=True,
            length=20,
            return_complex=False,
        )


def test_istft_decomposition_accumulates_overlapping_inverse_fft_frames():
    generator = torch.Generator().manual_seed(7)
    value = torch.complex(
        torch.randn((5, 7), generator=generator),
        torch.randn((5, 7), generator=generator),
    )
    _assert_compiled_matches(
        InverseShortTimeFourierTransform(), (value,), rtol=8e-4, atol=9e-5
    )
