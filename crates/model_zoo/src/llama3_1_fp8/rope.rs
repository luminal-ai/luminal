/// Host-side split-half RoPE tables with the Llama 3.1 frequency ramp.
///
/// Frequencies whose wavelength exceeds `original_max / low_freq_factor`
/// are divided by `factor`; wavelengths below
/// `original_max / high_freq_factor` retain their frequency; the band
/// between them is interpolated smoothly.
#[allow(clippy::too_many_arguments)]
pub fn rope_tables_llama3_scaled(
    positions: &[f32],
    head_dim: usize,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_positions: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let low_freq_wavelen = original_max_positions / low_freq_factor;
    let high_freq_wavelen = original_max_positions / high_freq_factor;
    let scaled_freq = |freq: f32| -> f32 {
        let wavelen = 2.0 * std::f32::consts::PI / freq;
        if wavelen > low_freq_wavelen {
            freq / factor
        } else if wavelen < high_freq_wavelen {
            freq
        } else {
            let smooth = (original_max_positions / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            (1.0 - smooth) * freq / factor + smooth * freq
        }
    };
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &position in positions {
        let (mut cos_row, mut sin_row) = (vec![0f32; head_dim], vec![0f32; head_dim]);
        for j in 0..half {
            let base = theta.powf(-2.0 * j as f32 / head_dim as f32);
            let arg = position * scaled_freq(base);
            cos_row[j] = arg.cos();
            cos_row[j + half] = arg.cos();
            sin_row[j] = arg.sin();
            sin_row[j + half] = arg.sin();
        }
        cos.extend(cos_row);
        sin.extend(sin_row);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::rope_tables_llama3_scaled;

    #[test]
    fn produces_full_width_split_half_rows() {
        let (cos, sin) =
            rope_tables_llama3_scaled(&[0.0, 1.0], 8, 500_000.0, 8.0, 1.0, 4.0, 8192.0);

        assert_eq!(cos.len(), 16);
        assert_eq!(sin.len(), 16);
        assert_eq!(&cos[..8], &[1.0; 8]);
        assert_eq!(&sin[..8], &[0.0; 8]);
        for index in 0..4 {
            assert_eq!(cos[8 + index], cos[12 + index]);
            assert_eq!(sin[8 + index], sin[12 + index]);
        }
    }
}
