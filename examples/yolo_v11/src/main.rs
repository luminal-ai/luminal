mod model;

use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use image::{ImageBuffer, ImageReader, Rgb, RgbImage};
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const ARTIFACT_DIR: &str = "examples/yolo_v11/artifacts";
const WEIGHTS_URL: &str =
    "https://github.com/luminal-ai/luminal/releases/download/yolo-v11n/weights.safetensors";
const SAMPLE_IMAGE_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg";
const CONF_THRES: f32 = 0.25;
const IOU_THRES: f32 = 0.45;
const MAX_DET: usize = 300;

#[derive(Debug, Clone, Copy)]
struct LetterboxMeta {
    orig_width: u32,
    orig_height: u32,
    ratio: f32,
    pad_x: f32,
    pad_y: f32,
}

#[derive(Debug, Clone)]
struct Detection {
    score: f32,
    class_id: usize,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, Clone)]
struct CliArgs {
    image_path: Option<PathBuf>,
    annotated_path: PathBuf,
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p yolo_v11 --bin yolo_v11 -- [--input <image.jpg|image.png>] [--output <annotated.png>]\n\
         \n\
         Positional form is also supported:\n\
         cargo run --release -p yolo_v11 --bin yolo_v11 -- <image.jpg|image.png> <annotated.png>\n\
         \n\
         If no image is supplied, the example uses examples/yolo_v11/artifacts/bus.jpg and downloads it if needed."
    );
}

fn cli_args(artifact_dir: &Path) -> CliArgs {
    let mut image_path = None;
    let mut annotated_path = None;
    let mut positionals = Vec::new();
    let mut args = env::args_os().skip(1);

    while let Some(arg) = args.next() {
        let arg_str = arg.to_string_lossy();
        match arg_str.as_ref() {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "--input" => {
                image_path = Some(next_cli_path(&mut args, arg_str.as_ref()));
            }
            "--output" | "-o" => {
                annotated_path = Some(next_cli_path(&mut args, arg_str.as_ref()));
            }
            "--" => {
                positionals.extend(args.map(PathBuf::from));
                break;
            }
            _ if arg_str.starts_with('-') => panic!("Unknown argument: {arg_str}"),
            _ => positionals.push(PathBuf::from(arg)),
        }
    }

    if let Some(positional) = positionals.first() {
        if image_path.is_some() {
            panic!("Input image was provided both positionally and with --input");
        }
        image_path = Some(positional.clone());
    }
    if let Some(positional) = positionals.get(1) {
        if annotated_path.is_some() {
            panic!("Output image was provided both positionally and with --output");
        }
        annotated_path = Some(positional.clone());
    }
    if positionals.len() > 2 {
        panic!("Too many positional arguments; expected at most <input> <output>");
    }

    let image_path = image_path.or_else(|| Some(artifact_dir.join("bus.jpg")));
    let annotated_path = annotated_path.unwrap_or_else(|| artifact_dir.join("annotated.png"));

    CliArgs {
        image_path,
        annotated_path,
    }
}

fn next_cli_path(args: &mut impl Iterator<Item = std::ffi::OsString>, flag: &str) -> PathBuf {
    args.next()
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{flag} requires a path"))
}

fn ensure_downloaded(path: &Path, url: &str, label: &str) {
    if path.exists() {
        return;
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Failed to create {}: {e}", parent.display()));
    }

    let tmp_path = download_temp_path(path);
    let _ = fs::remove_file(&tmp_path);

    println!("Downloading {label}: {url}");
    println!("  -> {}", path.display());

    let response = ureq::get(url)
        .set("User-Agent", "luminal-yolo-v11-example")
        .call()
        .unwrap_or_else(|e| panic!("Failed to download {label} from {url}: {e}"));
    let mut reader = response.into_reader();
    let mut file = fs::File::create(&tmp_path)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", tmp_path.display()));
    io::copy(&mut reader, &mut file)
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", tmp_path.display()));
    file.sync_all()
        .unwrap_or_else(|e| panic!("Failed to sync {}: {e}", tmp_path.display()));
    fs::rename(&tmp_path, path).unwrap_or_else(|e| {
        panic!(
            "Failed to move {} to {}: {e}",
            tmp_path.display(),
            path.display()
        )
    });
}

fn download_temp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    path.with_file_name(format!("{file_name}.download"))
}

fn preprocess_image(path: &Path) -> (Vec<f32>, LetterboxMeta) {
    let rgb = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode image {}: {e}", path.display()))
        .to_rgb8();
    let (orig_width, orig_height) = rgb.dimensions();
    assert!(
        orig_width > 0 && orig_height > 0,
        "image must have non-zero dimensions"
    );

    let img_size = IMG_SIZE as u32;
    let ratio = (img_size as f32 / orig_height as f32).min(img_size as f32 / orig_width as f32);
    let resized_width = ((orig_width as f32 * ratio).round() as u32).max(1);
    let resized_height = ((orig_height as f32 * ratio).round() as u32).max(1);
    let resized = resize_rgb_inter_linear(&rgb, resized_width, resized_height);

    let dw = img_size.saturating_sub(resized_width) as f32;
    let dh = img_size.saturating_sub(resized_height) as f32;
    let left = ((dw / 2.0) - 0.1).round().max(0.0) as u32;
    let top = ((dh / 2.0) - 0.1).round().max(0.0) as u32;

    let mut letterboxed: RgbImage =
        ImageBuffer::from_pixel(img_size, img_size, Rgb([114, 114, 114]));
    image::imageops::replace(&mut letterboxed, &resized, left.into(), top.into());

    let plane = IMG_SIZE * IMG_SIZE;
    let mut data = vec![0.0_f32; 3 * plane];
    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let p = letterboxed.get_pixel(x as u32, y as u32);
            let idx = y * IMG_SIZE + x;
            data[idx] = p[0] as f32 / 255.0;
            data[plane + idx] = p[1] as f32 / 255.0;
            data[2 * plane + idx] = p[2] as f32 / 255.0;
        }
    }

    (
        data,
        LetterboxMeta {
            orig_width,
            orig_height,
            ratio,
            pad_x: left as f32,
            pad_y: top as f32,
        },
    )
}

fn resize_rgb_inter_linear(src: &RgbImage, dst_width: u32, dst_height: u32) -> RgbImage {
    let (src_width, src_height) = src.dimensions();
    assert!(src_width > 0 && src_height > 0);
    if src_width == dst_width && src_height == dst_height {
        return src.clone();
    }

    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;
    let mut dst = RgbImage::new(dst_width, dst_height);

    for y in 0..dst_height {
        let (y0, y1, wy) = resize_axis(y, scale_y, src_height);
        for x in 0..dst_width {
            let (x0, x1, wx) = resize_axis(x, scale_x, src_width);
            let p00 = src.get_pixel(x0, y0);
            let p01 = src.get_pixel(x1, y0);
            let p10 = src.get_pixel(x0, y1);
            let p11 = src.get_pixel(x1, y1);
            let mut out = [0u8; 3];
            for c in 0..3 {
                let top = p00[c] as f32 * (1.0 - wx) + p01[c] as f32 * wx;
                let bottom = p10[c] as f32 * (1.0 - wx) + p11[c] as f32 * wx;
                out[c] = (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
            }
            dst.put_pixel(x, y, Rgb(out));
        }
    }

    dst
}

fn resize_axis(dst_index: u32, scale: f32, src_len: u32) -> (u32, u32, f32) {
    if src_len == 1 {
        return (0, 0, 0.0);
    }

    let src = (dst_index as f32 + 0.5) * scale - 0.5;
    if src < 0.0 {
        return (0, 0, 0.0);
    }

    let mut i0 = src.floor() as u32;
    let mut weight = src - i0 as f32;
    if i0 >= src_len - 1 {
        i0 = src_len - 2;
        weight = 1.0;
    }
    (i0, i0 + 1, weight)
}

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let cwd = std::env::current_dir().unwrap();
    let artifact_dir = cwd.join(ARTIFACT_DIR);
    let weights_path = artifact_dir.join("weights.safetensors");
    let cli = cli_args(&artifact_dir);
    let image_path = cli.image_path.clone();
    let search_graphs = 50usize;

    println!("Using artifact directory: {}", artifact_dir.display());

    ensure_downloaded(&weights_path, WEIGHTS_URL, "YOLO v11n Luminal weights");

    let image_path = image_path.unwrap_or_else(|| {
        panic!(
            "No input image supplied and default image is missing; pass --input <image.jpg|image.png>"
        )
    });
    if image_path == artifact_dir.join("bus.jpg") {
        ensure_downloaded(&image_path, SAMPLE_IMAGE_URL, "sample image");
    }
    assert!(
        image_path.exists(),
        "Image path does not exist: {}",
        image_path.display()
    );
    println!("Input image: {}", image_path.display());
    let (img_data, letterbox_meta) = preprocess_image(&image_path);
    println!(
        "  original={}x{} letterbox_ratio={:.6} pad=({:.0}, {:.0})",
        letterbox_meta.orig_width,
        letterbox_meta.orig_height,
        letterbox_meta.ratio,
        letterbox_meta.pad_x,
        letterbox_meta.pad_y
    );
    let expected_input = 3 * IMG_SIZE * IMG_SIZE;
    assert_eq!(img_data.len(), expected_input, "input size mismatch");

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Build graph
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let yolo = YoloV11::init(&mut cx);
    let logits = yolo.forward(img).output();

    println!("Building E-Graph...");
    let t0 = Instant::now();
    cx.build_search_space::<CudaRuntime>();
    println!("  built E-Graph in {:?}", t0.elapsed());

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    // Initialize anchors, strides, and DFL constant.
    let (anchors_flat, strides_flat) = make_anchors_and_strides(&[80, 40, 20], &STRIDES);
    runtime.set_data(yolo.detect.anchors, anchors_flat.clone());
    runtime.set_data(yolo.detect.strides, strides_flat.clone());
    runtime.set_data(yolo.detect.dfl_weight, dfl_weight());

    runtime.set_data(img, img_data.clone());

    println!("Compiling (search_graphs={search_graphs})...");
    let t0 = Instant::now();
    runtime = cx.search(runtime, search_graphs);
    println!("  search took {:?}", t0.elapsed());

    // Re-set anchors/strides/dfl/img after search (search may consume the inputs)
    runtime.set_data(yolo.detect.anchors, anchors_flat);
    runtime.set_data(yolo.detect.strides, strides_flat);
    runtime.set_data(yolo.detect.dfl_weight, dfl_weight());
    runtime.set_data(img, img_data);

    println!("Executing...");
    let t0 = Instant::now();
    runtime.execute(&cx.dyn_map);
    let elapsed = t0.elapsed();
    println!("  forward took {:?}", elapsed);

    // Get output (1, 4 + NC, 8400) — Detect with export=True returns the
    // DECODED predictions (4 box coords + NC class scores), not the raw
    // (NC + REG_MAX*4) channels.
    let out = runtime.get_f32(logits);
    let total_anchors: usize = 80 * 80 + 40 * 40 + 20 * 20;
    let expected_out_len = (4 + NC) * total_anchors;
    println!(
        "  output buffer length: {} (expected {} for shape (1, {}, {}))",
        out.len(),
        expected_out_len,
        4 + NC,
        total_anchors
    );
    let out = &out[..expected_out_len];

    let detections = nms_detections(out, total_anchors, CONF_THRES, IOU_THRES, MAX_DET);
    print_detections(&detections, Some(letterbox_meta));

    save_annotated_image(
        &image_path,
        &cli.annotated_path,
        &detections,
        letterbox_meta,
    );
    println!("Wrote annotated image: {}", cli.annotated_path.display());
}

fn print_detections(detections: &[Detection], meta: Option<LetterboxMeta>) {
    println!(
        "Detections after NMS (conf >= {:.2}, iou <= {:.2}):",
        CONF_THRES, IOU_THRES
    );
    if detections.is_empty() {
        println!("  none");
        return;
    }

    let coco_names = coco_names();
    for det in detections.iter().take(20) {
        let name = coco_names.get(det.class_id).copied().unwrap_or("?");
        let (x1, y1, x2, y2) = if let Some(meta) = meta {
            map_to_original(det.x1, det.y1, det.x2, det.y2, meta)
        } else {
            (det.x1, det.y1, det.x2, det.y2)
        };
        println!(
            "  conf={:.3} class={:>14}  xyxy=[{:.1}, {:.1}, {:.1}, {:.1}]",
            det.score, name, x1, y1, x2, y2
        );
    }
}

fn save_annotated_image(
    input_path: &Path,
    output_path: &Path,
    detections: &[Detection],
    meta: LetterboxMeta,
) {
    let mut image = ImageReader::open(input_path)
        .unwrap_or_else(|e| panic!("Failed to open image {}: {e}", input_path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode image {}: {e}", input_path.display()))
        .to_rgb8();
    let names = coco_names();
    let thickness = ((image.width().min(image.height()) as f32 / 320.0).round() as u32).max(2);

    for det in detections.iter().take(MAX_DET) {
        let (x1, y1, x2, y2) = map_to_original(det.x1, det.y1, det.x2, det.y2, meta);
        let color = class_color(det.class_id);
        draw_rect(&mut image, x1, y1, x2, y2, color, thickness);
        let name = names.get(det.class_id).copied().unwrap_or("?");
        draw_label(
            &mut image,
            x1,
            y1,
            &format!("{name} {:.2}", det.score),
            color,
        );
    }

    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("Failed to create {}: {e}", parent.display()));
    }
    image.save(output_path).unwrap_or_else(|e| {
        panic!(
            "Failed to write annotated image {}: {e}",
            output_path.display()
        )
    });
}

fn class_color(class_id: usize) -> Rgb<u8> {
    const COLORS: [[u8; 3]; 20] = [
        [220, 38, 38],
        [37, 99, 235],
        [22, 163, 74],
        [217, 119, 6],
        [147, 51, 234],
        [8, 145, 178],
        [219, 39, 119],
        [101, 163, 13],
        [234, 88, 12],
        [79, 70, 229],
        [15, 118, 110],
        [190, 18, 60],
        [124, 58, 237],
        [202, 138, 4],
        [2, 132, 199],
        [132, 204, 22],
        [249, 115, 22],
        [168, 85, 247],
        [20, 184, 166],
        [244, 63, 94],
    ];
    Rgb(COLORS[class_id % COLORS.len()])
}

fn draw_rect(
    image: &mut RgbImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Rgb<u8>,
    thickness: u32,
) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }

    let left = x1.min(x2).floor().clamp(0.0, (width - 1) as f32) as u32;
    let right = x1.max(x2).ceil().clamp(0.0, (width - 1) as f32) as u32;
    let top = y1.min(y2).floor().clamp(0.0, (height - 1) as f32) as u32;
    let bottom = y1.max(y2).ceil().clamp(0.0, (height - 1) as f32) as u32;
    if left > right || top > bottom {
        return;
    }

    for t in 0..thickness {
        if top + t <= bottom {
            draw_hline(image, left, right, top + t, color);
        }
        if bottom >= t && bottom - t >= top {
            draw_hline(image, left, right, bottom - t, color);
        }
        if left + t <= right {
            draw_vline(image, left + t, top, bottom, color);
        }
        if right >= t && right - t >= left {
            draw_vline(image, right - t, top, bottom, color);
        }
    }
}

fn draw_hline(image: &mut RgbImage, x1: u32, x2: u32, y: u32, color: Rgb<u8>) {
    if y >= image.height() {
        return;
    }
    let start = x1.min(x2).min(image.width().saturating_sub(1));
    let end = x1.max(x2).min(image.width().saturating_sub(1));
    for x in start..=end {
        image.put_pixel(x, y, color);
    }
}

fn draw_vline(image: &mut RgbImage, x: u32, y1: u32, y2: u32, color: Rgb<u8>) {
    if x >= image.width() {
        return;
    }
    let start = y1.min(y2).min(image.height().saturating_sub(1));
    let end = y1.max(y2).min(image.height().saturating_sub(1));
    for y in start..=end {
        image.put_pixel(x, y, color);
    }
}

fn draw_label(image: &mut RgbImage, box_x: f32, box_y: f32, text: &str, color: Rgb<u8>) {
    let scale = ((image.width().min(image.height()) as f32 / 500.0).round() as u32).max(2);
    let text = text.to_ascii_uppercase();
    let text_width = text_pixel_width(&text, scale);
    let text_height = 7 * scale;
    let pad = 3 * scale;
    let label_width = text_width + pad * 2;
    let label_height = text_height + pad * 2;

    let mut x = box_x.floor().max(0.0) as u32;
    if x + label_width >= image.width() {
        x = image.width().saturating_sub(label_width + 1);
    }

    let box_top = box_y.floor().max(0.0) as u32;
    let y = if box_top > label_height {
        box_top - label_height
    } else {
        box_top.min(image.height().saturating_sub(label_height + 1))
    };

    fill_rect(image, x, y, label_width, label_height, color);
    draw_text_5x7(image, x + pad, y + pad, &text, Rgb([255, 255, 255]), scale);
}

fn fill_rect(image: &mut RgbImage, x: u32, y: u32, width: u32, height: u32, color: Rgb<u8>) {
    let max_x = (x + width).min(image.width());
    let max_y = (y + height).min(image.height());
    for py in y..max_y {
        for px in x..max_x {
            image.put_pixel(px, py, color);
        }
    }
}

fn text_pixel_width(text: &str, scale: u32) -> u32 {
    let mut width = 0;
    for ch in text.chars() {
        width += if ch == ' ' { 3 * scale } else { 5 * scale };
        width += scale;
    }
    width.saturating_sub(scale)
}

fn draw_text_5x7(image: &mut RgbImage, x: u32, y: u32, text: &str, color: Rgb<u8>, scale: u32) {
    let mut cursor = x;
    for ch in text.chars() {
        if ch == ' ' {
            cursor += 4 * scale;
            continue;
        }
        draw_glyph_5x7(image, cursor, y, ch, color, scale);
        cursor += 6 * scale;
    }
}

fn draw_glyph_5x7(image: &mut RgbImage, x: u32, y: u32, ch: char, color: Rgb<u8>, scale: u32) {
    let Some(rows) = glyph_5x7(ch) else {
        return;
    };
    for (row_idx, row) in rows.iter().enumerate() {
        for (col_idx, pixel) in row.as_bytes().iter().enumerate() {
            if *pixel != b'1' {
                continue;
            }
            let px = x + col_idx as u32 * scale;
            let py = y + row_idx as u32 * scale;
            fill_rect(image, px, py, scale, scale, color);
        }
    }
}

fn glyph_5x7(ch: char) -> Option<[&'static str; 7]> {
    Some(match ch {
        'A' => [
            "01110", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'B' => [
            "11110", "10001", "10001", "11110", "10001", "10001", "11110",
        ],
        'C' => [
            "01111", "10000", "10000", "10000", "10000", "10000", "01111",
        ],
        'D' => [
            "11110", "10001", "10001", "10001", "10001", "10001", "11110",
        ],
        'E' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "11111",
        ],
        'F' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "10000",
        ],
        'G' => [
            "01111", "10000", "10000", "10011", "10001", "10001", "01111",
        ],
        'H' => [
            "10001", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'I' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "11111",
        ],
        'J' => [
            "00111", "00010", "00010", "00010", "00010", "10010", "01100",
        ],
        'K' => [
            "10001", "10010", "10100", "11000", "10100", "10010", "10001",
        ],
        'L' => [
            "10000", "10000", "10000", "10000", "10000", "10000", "11111",
        ],
        'M' => [
            "10001", "11011", "10101", "10101", "10001", "10001", "10001",
        ],
        'N' => [
            "10001", "11001", "10101", "10011", "10001", "10001", "10001",
        ],
        'O' => [
            "01110", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'P' => [
            "11110", "10001", "10001", "11110", "10000", "10000", "10000",
        ],
        'Q' => [
            "01110", "10001", "10001", "10001", "10101", "10010", "01101",
        ],
        'R' => [
            "11110", "10001", "10001", "11110", "10100", "10010", "10001",
        ],
        'S' => [
            "01111", "10000", "10000", "01110", "00001", "00001", "11110",
        ],
        'T' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "00100",
        ],
        'U' => [
            "10001", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'V' => [
            "10001", "10001", "10001", "10001", "10001", "01010", "00100",
        ],
        'W' => [
            "10001", "10001", "10001", "10101", "10101", "10101", "01010",
        ],
        'X' => [
            "10001", "10001", "01010", "00100", "01010", "10001", "10001",
        ],
        'Y' => [
            "10001", "10001", "01010", "00100", "00100", "00100", "00100",
        ],
        'Z' => [
            "11111", "00001", "00010", "00100", "01000", "10000", "11111",
        ],
        '0' => [
            "01110", "10001", "10011", "10101", "11001", "10001", "01110",
        ],
        '1' => [
            "00100", "01100", "00100", "00100", "00100", "00100", "01110",
        ],
        '2' => [
            "01110", "10001", "00001", "00010", "00100", "01000", "11111",
        ],
        '3' => [
            "11110", "00001", "00001", "01110", "00001", "00001", "11110",
        ],
        '4' => [
            "00010", "00110", "01010", "10010", "11111", "00010", "00010",
        ],
        '5' => [
            "11111", "10000", "10000", "11110", "00001", "00001", "11110",
        ],
        '6' => [
            "01110", "10000", "10000", "11110", "10001", "10001", "01110",
        ],
        '7' => [
            "11111", "00001", "00010", "00100", "01000", "01000", "01000",
        ],
        '8' => [
            "01110", "10001", "10001", "01110", "10001", "10001", "01110",
        ],
        '9' => [
            "01110", "10001", "10001", "01111", "00001", "00001", "01110",
        ],
        '.' => [
            "00000", "00000", "00000", "00000", "00000", "01100", "01100",
        ],
        '-' => [
            "00000", "00000", "00000", "11111", "00000", "00000", "00000",
        ],
        '/' => [
            "00001", "00010", "00010", "00100", "01000", "01000", "10000",
        ],
        '?' => [
            "01110", "10001", "00001", "00010", "00100", "00000", "00100",
        ],
        _ => return None,
    })
}

fn nms_detections(
    out: &[f32],
    total_anchors: usize,
    conf_thres: f32,
    iou_thres: f32,
    max_det: usize,
) -> Vec<Detection> {
    let nc = NC;
    let mut candidates = Vec::new();
    for a in 0..total_anchors {
        let cx = out[a];
        let cy = out[total_anchors + a];
        let w = out[2 * total_anchors + a];
        let h = out[3 * total_anchors + a];
        let mut best_score = 0.0_f32;
        let mut best_class = 0usize;
        for c in 0..nc {
            let s = out[(4 + c) * total_anchors + a];
            if s > best_score {
                best_score = s;
                best_class = c;
            }
        }
        if best_score >= conf_thres {
            candidates.push(Detection {
                score: best_score,
                class_id: best_class,
                x1: cx - w / 2.0,
                y1: cy - h / 2.0,
                x2: cx + w / 2.0,
                y2: cy + h / 2.0,
            });
        }
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep: Vec<Detection> = Vec::new();
    'candidate: for candidate in candidates {
        for selected in &keep {
            if candidate.class_id == selected.class_id && box_iou(&candidate, selected) > iou_thres
            {
                continue 'candidate;
            }
        }
        keep.push(candidate);
        if keep.len() >= max_det {
            break;
        }
    }
    keep
}

fn box_iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let intersection = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let a_area = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let b_area = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    intersection / (a_area + b_area - intersection + f32::EPSILON)
}

fn map_to_original(
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    meta: LetterboxMeta,
) -> (f32, f32, f32, f32) {
    let ox1 = ((x1 - meta.pad_x) / meta.ratio).clamp(0.0, meta.orig_width as f32);
    let oy1 = ((y1 - meta.pad_y) / meta.ratio).clamp(0.0, meta.orig_height as f32);
    let ox2 = ((x2 - meta.pad_x) / meta.ratio).clamp(0.0, meta.orig_width as f32);
    let oy2 = ((y2 - meta.pad_y) / meta.ratio).clamp(0.0, meta.orig_height as f32);
    (ox1, oy1, ox2, oy2)
}

fn coco_names() -> [&'static str; NC] {
    [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
}
