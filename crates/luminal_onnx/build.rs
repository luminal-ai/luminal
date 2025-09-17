fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure protoc is available without a system install.
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);

    let mut config = prost_build::Config::new();
    // Use BTreeMap for determinism where prost would generate HashMaps.
    config.btree_map(["."]);
    prost_build::compile_protos(&["src/onnx/proto/onnx.proto"], &["src/onnx/proto/"])?;
    Ok(())
}
