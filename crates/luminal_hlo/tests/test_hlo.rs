use luminal_hlo::import_hlo;

luminal::test_imports!();

#[test]
fn test_stablehlo_unary_ops() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_unary_ops.mlir");

    inputs["%arg0"].set([[2401., 4096.], [625., 1296.]]);

    cx.execute();

    let expected = [1. / 7., 1. / 8.];
    assert_close(&inputs["%7"].data(), &expected);
}

#[test]
fn test_stablehlo_binary_ops() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_binary_ops.mlir");

    inputs["%arg0"].set([[9., 8.], [7., 6.]]);
    inputs["%arg1"].set([[2., 2.], [3., 4.]]);

    cx.execute();

    let expected = [2., 2., 3., 4., 2., 2., 3., 4.];
    assert_close(&inputs["%8"].data(), &expected);
}

#[test]
fn test_stablehlo_ternary_ops() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_ternary_ops.mlir");

    inputs["%arg0"].set([0., 1., 0., 1., 0.]);
    inputs["%arg1"].set([10., 20., 30., 40., 50.]);

    cx.execute();

    let expected = [1., 20., 3., 40., 5.];
    assert_close(&inputs["%1"].data(), &expected);
}

#[test]
fn test_stablehlo_constant_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_constant_op.mlir");

    inputs["%arg0"].set([1., 1., 1., 1.]);

    cx.execute();

    let expected = [169., 169., 169., 169.];
    assert_close(&inputs["%2"].data(), &expected);
}

#[test]
fn test_stablehlo_broadcast_in_dim_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_broadcast_in_dim_op.mlir");

    inputs["%arg0"].set([1., 1., 1., 1.]);
    inputs["%arg1"].set([1., 1., 1., 1.]);

    cx.execute();

    let expected = [20., 20., 20., 20., 20., 20., 20., 20.];
    assert_close(&inputs["%2"].data(), &expected);
}

#[test]
fn test_stablehlo_convolution_1x1_nchw() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_convolution_1x1_nchw.mlir");

    inputs["%arg0"].set([1.0, 2.0, 3.0, 4.0]);
    inputs["%arg1"].set([2.0]);

    cx.execute();

    let expected = [2.0, 4.0, 6.0, 8.0];
    assert_close(&inputs["%0"].data(), &expected);
}

#[test]
fn test_stablehlo_convolution_3x3_nchw() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_convolution_3x3_nchw.mlir");

    inputs["%arg0"].set([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    inputs["%arg1"].set([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    cx.execute();

    let expected = [25.0];
    assert_close(&inputs["%0"].data(), &expected);
}

#[test]
fn test_stablehlo_reduce_window() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_reduce_window.mlir");

    inputs["%arg0"].set([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]);

    cx.execute();

    let expected = [6.0, 8.0, 14.0, 16.0];
    assert_close(&inputs["%0"].data(), &expected);
}

#[test]
fn test_stablehlo_compare_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_compare_op.mlir");

    inputs["%arg0"].set([1., 2., 3., 4., 5.]);
    inputs["%arg1"].set([1., 2., 3., 4., 5.]);

    cx.execute();

    // Concatenates the results of comparison ops in the following order: NE, GT, GE, LT, LE, EQ
    let expected = [
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1.,
    ];
    assert_close(&inputs["%6"].data(), &expected);
}

#[test]
fn test_stablehlo_logical_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_logical_op.mlir");

    inputs["%arg0"].set([0., 1., 0., 1., 0.]);

    cx.execute();

    let expected = [1., 0., 1., 0., 1.];
    assert_close(&inputs["%0"].data(), &expected);
}

#[test]
fn test_stablehlo_dot_general_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_dot_general_op.mlir");

    inputs["%arg0"].set([1., 2., 3., 4., 5., 6.]);
    inputs["%arg1"].set([7., 8., 9., 10., 11., 12.]);

    cx.execute();

    let expected = [58., 64., 139., 154.];
    assert_close(&inputs["%0"].data(), &expected);
}

#[test]
fn test_stablehlo_dot_general_op_batching() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_dot_general_op_batching.mlir");

    inputs["%arg0"].set([1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 0., 1.]);
    inputs["%arg1"].set([1., 0., 0., 1., 1., 1., 1., 2., 3., 4., 5., 6.]);

    cx.execute();

    let expected = [4., 5., 10., 11., 76., 100., 6., 8.];
    assert_close(&inputs["%0"].data(), &expected);
}

#[test]
fn test_stablehlo_reduce_op() {
    let (mut cx, inputs) = import_hlo("tests/data/stablehlo_reduce_op.mlir");

    inputs["%arg0"].set([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, -1.0, 2.0, 7.0, 7.0, -3.0]);
    inputs["%arg1"].set([0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.]);

    cx.execute();

    let expected = [6., 3., 1., 15., 6., 0., 1., 2., 1., 11., 7., 1.];
    assert_close(&inputs["%6"].data(), &expected);
}
