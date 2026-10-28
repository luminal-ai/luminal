"""Known failures of the PT2 parity suites on the CPU reference backend.

Every entry below failed on main in the Test Python Native job as of
2026-10-26 (run 40050394623). They are grouped by root cause, and each
group carries the Linear ticket that owns it. The conftests of both test
trees apply these as ``pytest.mark.xfail`` ONLY when the selected backend
is ``reference``; the CUDA job is unaffected.

Entries are keyed by ``<test file basename>::<test name with parameters>``
so they match regardless of the rootdir pytest resolves. Remove a test from
its group (or the whole group) when the ticket is fixed; a passing entry
shows up as XPASS, which is harmless but should be cleaned up.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import backend_test_utils
import pytest


@dataclass(frozen=True)
class KnownFailure:
    ticket: str
    url: str
    reason: str
    tests: tuple[str, ...]
    # ``platform.machine()`` values the failure is known on; empty means
    # every machine. A machine-specific entry runs as an ordinary test
    # elsewhere, so it keeps asserting where it is known to pass.
    machines: tuple[str, ...] = ()

    def applies_here(self) -> bool:
        return not self.machines or platform.machine() in self.machines

    @property
    def xfail_reason(self) -> str:
        return f"{self.ticket}: {self.reason}. {self.url}"


REFERENCE_KNOWN_FAILURES: tuple[KnownFailure, ...] = (
    KnownFailure(
        ticket="LUM-1100",
        url="https://linear.app/luminalai/issue/LUM-1100/complex-tensor-inputs-fail-with-input-rank-differs-from-exported-rank",
        reason=(
            "complex inputs fail set_input with 'input rank differs from exported rank'"
        ),
        tests=(
            "test_complex.py::test_abs_and_division_avoid_naive_overflow_and_underflow",
            "test_complex.py::test_complex32_boundary_uses_f16_components",
            "test_complex.py::test_complex_asin_asinh_avoid_cancellation",
            "test_complex.py::test_complex_buffer_uses_component_storage_at_weight_boundary",
            "test_complex.py::test_complex_components_and_shape_operations",
            "test_complex.py::test_complex_predicate_aliases_lower_through_real_components",
            "test_complex.py::test_complex_scalar_casts_and_noncontiguous_negative_views",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[exp2]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[expm1]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[log10]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[log1p]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[log2]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[log]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[reciprocal]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[rsqrt]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[sigmoid]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[sin]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[sinh]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[sqrt]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[tan]",
            "test_complex.py::test_complex_unary_aliases_lower_through_real_components[tanh]",
            "test_complex.py::test_mixed_real_complex_and_real_scalars",
            "test_constructors.py::test_full_like_supports_nonzero_complex_fill",
            "test_straightforward_lowerings.py::test_addmv_and_addbmm_support_complex_coefficients",
        ),
    ),
    KnownFailure(
        ticket="LUM-1101",
        url="https://linear.app/luminalai/issue/LUM-1101/complex-aten-ops-are-not-lowered-through-real-components-fft-prod",
        reason=("complex ATen op is not lowered through real components"),
        tests=(
            "test_complex.py::test_complex_arithmetic_reductions_and_matmul[dtype0-0.0002-2e-05]",
            "test_complex.py::test_complex_arithmetic_reductions_and_matmul[dtype1-1e-10-1e-12]",
            "test_complex.py::test_complex_copy_aliases_materialize_every_output",
            "test_complex.py::test_complex_empty_dimensions",
            "test_complex.py::test_complex_power_and_scale_aliases_lower_through_real_components",
            "test_complex.py::test_complex_scatter_aliases_preserve_duplicate_updates",
            "test_complex.py::test_complex_split_with_sizes_preserves_all_outputs[dtype0]",
            "test_complex.py::test_complex_split_with_sizes_preserves_all_outputs[dtype1]",
            "test_complex.py::test_complex_split_with_sizes_preserves_all_outputs[dtype2]",
            "test_constructors.py::test_constant_pad_nd_preserves_typed_fill_and_ieee_values",
            "test_fft.py::test_c2r_ignores_dc_and_nyquist_imaginary_components",
            "test_fft.py::test_complex_one_dimensional_fft_family_even_and_odd_lengths",
            "test_fft.py::test_double_precision_fft_accuracy_and_dtype",
            "test_fft.py::test_fft_accepts_symbolic_compile_time_length_expression",
            "test_fft.py::test_fft_shifts_for_integer_and_complex_values",
            "test_fft.py::test_istft_decomposition_accumulates_overlapping_inverse_fft_frames",
            "test_fft.py::test_multidimensional_complex_inverse_and_hermitian_ffts",
            "test_fft.py::test_multidimensional_fft_family_and_nonfinal_axes",
            "test_fft.py::test_real_one_dimensional_fft_family",
            "test_reduction.py::test_composed_complex_reductions_match_eager",
            "test_reduction.py::test_cumprod_complex_scalar_and_empty_inputs",
            "test_remaining_existing_hlir_lowerings.py::test_complex_movement_is_componentwise",
            "test_straightforward_lowerings.py::test_copy_supports_broadcast_and_real_complex_conversions",
            "test_straightforward_lowerings.py::test_optional_index_put_accumulates_real_and_complex_duplicates",
        ),
    ),
    KnownFailure(
        ticket="LUM-1097",
        url="https://linear.app/luminalai/issue/LUM-1097/layoutir-extraction-dead-ends-on-int32int64-and-empty-input-graphs",
        reason=(
            "LayoutIR extraction dead-ends on int32/int64 and empty-input graphs (outputs with no plan)"
        ),
        tests=(
            "test_cpu_staging.py::test_empty_cpu_input_preserves_shape_and_dtype[int32]",
            "test_cpu_staging.py::test_empty_cpu_input_preserves_shape_and_dtype[int64]",
            "test_dtype_boundary.py::test_boundary_noop_preserves_dtype_and_values[int32]",
            "test_dtype_boundary.py::test_boundary_noop_preserves_dtype_and_values[int64_i32_range]",
            "test_dtype_boundary.py::test_boundary_noop_preserves_dtype_and_values[int64_outside_i32_range]",
            "test_dtype_boundary.py::test_matching_dtype_does_not_raise[int32]",
            "test_dtype_boundary.py::test_matching_dtype_does_not_raise[int64_i32_range]",
            "test_dtype_boundary.py::test_matching_dtype_does_not_raise[int64_outside_i32_range]",
            "test_hlir_ops.py::test_reduce_sum_all_axes_int64_preserves_dtype",
            "test_ops_batch5.py::test_upsample_bilinear",
            "test_remaining_existing_hlir_lowerings.py::test_antialiased_bilinear_resize_float_and_uint8",
            "test_remaining_existing_hlir_lowerings.py::test_integer_gcd_static_euclidean_rounds",
            "test_scalars.py::test_dot_float_shorthand_on_0d",
            "test_scalars.py::test_int_sum_produces_int_scalar",
            "test_writeback_mutations.py::test_static_cache_prefill_smoke",
            "test_writeback_mutations.py::test_writeback_metadata_exposed",
        ),
    ),
    KnownFailure(
        ticket="LUM-1098",
        url="https://linear.app/luminalai/issue/LUM-1098/reference-scattergather-executor-rejects-valid-graphs-dtype-agreement",
        reason=(
            "reference scatter/gather executor rejects the graph (dtype-agreement false positive, injectivity, gather bounds)"
        ),
        tests=(
            "test_constructors.py::test_full_preserves_dtype_and_literal_value",
            "test_constructors.py::test_scalar_tensor_is_typed_zero_dimensional",
            "test_gather_scatter_dynamic.py::test_scatter_elements_dynamic_index_shape",
            "test_gather_scatter_dynamic.py::test_scatter_nd_dynamic_data_shape",
            "test_hlir_ops.py::test_int_index_put_scalar_src",
            "test_ops_batch4.py::test_masked_scatter",
            "test_ops_batch4.py::test_scatter",
            "test_straightforward_lowerings.py::test_movement_lowerings_support_noncontiguous_real_and_complex_values",
        ),
    ),
    KnownFailure(
        ticket="LUM-1102",
        url="https://linear.app/luminalai/issue/LUM-1102/referenceaotbackend-does-not-accept-the-export-mode-option-used-by",
        reason=("ReferenceAOTBackend does not accept the export_mode option"),
        tests=(
            "test_hlir_ops.py::test_conv1d_floor_div_positional_pt2",
            "test_hlir_ops.py::test_conv1d_no_pad_pt2",
            "test_hlir_ops.py::test_conv2d_dilation_pt2",
            "test_hlir_ops.py::test_conv2d_no_pad_pt2",
            "test_hlir_ops.py::test_conv3d_same_pad_pt2",
            "test_hlir_ops.py::test_depthwise_multiplier_conv2d_pt2",
            "test_hlir_ops.py::test_grouped_conv2d_groups3_batch4_pt2",
        ),
    ),
    KnownFailure(
        ticket="LUM-1103",
        url="https://linear.app/luminalai/issue/LUM-1103/runtime-scalar-outputs-item-local-scalar-dense-unbacked-symints",
        reason=(
            "runtime scalar outputs (item / _local_scalar_dense / unbacked symints) are unsupported on the reference AOT path"
        ),
        tests=(
            "test_constructors.py::test_constructors_accept_tensor_backed_runtime_scalars",
            "test_dtype_boundary.py::test_item_returns_exact_python_scalar[bfloat16]",
            "test_dtype_boundary.py::test_item_returns_exact_python_scalar[bool]",
            "test_dtype_boundary.py::test_item_returns_exact_python_scalar[float16]",
            "test_dtype_boundary.py::test_item_returns_exact_python_scalar[float32]",
            "test_dtype_boundary.py::test_item_returns_exact_python_scalar[float64-precision-sensitive]",
            "test_dtype_boundary.py::test_item_returns_exact_python_scalar[int32]",
            "test_dtype_boundary.py::test_item_returns_exact_python_scalar[int64]",
        ),
    ),
    KnownFailure(
        ticket="LUM-1104",
        url="https://linear.app/luminalai/issue/LUM-1104/pt2-translator-gaps-arange-non-integer-start-full-with-unresolved-sym",
        reason=("PT2 translator does not port this ATen variant yet"),
        tests=(
            "test_constructors.py::test_full_supports_symbolic_shapes",
            "test_dtype_boundary.py::test_arange_uses_exported_shape_and_declared_dtype[bool-scalars]",
            "test_dtype_boundary.py::test_arange_uses_exported_shape_and_declared_dtype[empty]",
            "test_dtype_boundary.py::test_arange_uses_exported_shape_and_declared_dtype[fractional-to-int]",
            "test_fft.py::test_stft_decomposition_uses_composed_r2c_lowering",
            "test_hlir_ops.py::test_onehot",
            "test_hlir_ops.py::test_sdpa_gqa_causal_bf16_parity",
            "test_hlir_ops.py::test_sdpa_gqa_fp32",
            "test_reduction.py::test_scatter_and_index_reduce_accept_bounded_symbolic_update_extents",
            "test_remaining_existing_hlir_lowerings.py::test_elementwise_composites_cover_ieee_edges",
            "test_straightforward_lowerings.py::test_index_select_and_unfold_match_scalar_semantics",
            "test_straightforward_lowerings.py::test_small_algebraic_aliases_use_existing_primitives",
        ),
    ),
    KnownFailure(
        ticket="LUM-1105",
        url="https://linear.app/luminalai/issue/LUM-1105/reference-backend-numeric-mismatches-vs-eager-float64-add-alpha-acos",
        reason=("reference backend output differs numerically from eager"),
        tests=(
            "test_hlir_ops.py::test_gelu_tanh",
            "test_reduction.py::test_scatter_reduce_integer_mean_floors_negative_results",
            "test_remaining_existing_hlir_lowerings.py::test_ordering_and_distance_composites",
            "test_straightforward_lowerings.py::test_addmv_and_addbmm_preserve_real_dtypes_and_coefficients",
            "test_upsample.py::test_upsample_nearest_noninteger_scale_same_size",
        ),
    ),
    KnownFailure(
        ticket="LUM-1106",
        url="https://linear.app/luminalai/issue/LUM-1106/ieee-signed-zero-and-nan-semantics-on-the-reference-backend-atan2",
        reason=(
            "signed-zero / NaN results differ on the Linux x86-64 CI runner "
            "(fmax(-0.0, -0.0) returns +0.0 there); every case passes on arm64 "
            "macOS in debug and release builds, and the x86 cause is not yet isolated"
        ),
        tests=(
            "test_binary.py::test_copysign_fmax_fmin_nan_and_zero_signs",
            "test_binary.py::test_fmax_fmin_skip_nan_and_order_signed_zeros[dtype0]",
            "test_binary.py::test_fmax_fmin_skip_nan_and_order_signed_zeros[dtype1]",
            "test_binary.py::test_fmax_fmin_skip_nan_and_order_signed_zeros[dtype2]",
            "test_binary.py::test_fmax_fmin_skip_nan_and_order_signed_zeros[dtype3]",
            "test_complex.py::test_acos_and_acosh_take_the_side_of_the_cut_from_a_signed_zero[torch.float64]",
            "test_remaining_existing_hlir_lowerings.py::test_nanmedian_slot_matches_eager_on_the_backend_path",
            "test_unary.py::test_erf_family_is_odd_through_an_exact_sign[gelu]",
        ),
        machines=("x86_64", "AMD64"),
    ),
    KnownFailure(
        ticket="LUM-1107",
        url="https://linear.app/luminalai/issue/LUM-1107/frontend-shape-assertions-panic-through-pyo3-instead-of-returning",
        reason=(
            "frontend shape assertion panics through pyo3 instead of returning a translation error"
        ),
        tests=(
            "test_hlir_ops.py::test_repeat_leading_and_tile",
            "test_hlir_ops.py::test_repeat_leading_dims",
            "test_llama3.py::test_hf_llama3_1b_bf16_error_within_fp32_floor",
            "test_reduction.py::test_scalar_log_softmax_preserves_ieee_behavior",
            "test_tiny_llm.py::test_tiny_llm_dynamic_context_reuses_one_compile",
        ),
    ),
    KnownFailure(
        ticket="LUM-1108",
        url="https://linear.app/luminalai/issue/LUM-1108/multi-process-collective-and-spmd-tests-are-flaky-on-the-python-native",
        reason=(
            "multi-process collective/SPMD test is flaky on the CI runner (timeout or one-element mismatch)"
        ),
        tests=(
            "test_collectives.py::test_three_rank_collectives",
            "test_spmd.py::test_two_rank_dtensor_training",
        ),
    ),
)


def _key(item: pytest.Item) -> str:
    return f"{item.path.name}::{item.name}"


def apply_reference_xfails(items: list[pytest.Item]) -> None:
    """Mark every known reference-backend failure as xfail (non-strict)."""
    if backend_test_utils.selected_backend() != "reference":
        return
    reasons = {
        test: failure.xfail_reason
        for failure in REFERENCE_KNOWN_FAILURES
        if failure.applies_here()
        for test in failure.tests
    }
    for item in items:
        reason = reasons.get(_key(item))
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
