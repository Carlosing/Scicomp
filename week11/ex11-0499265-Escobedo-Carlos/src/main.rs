use num::Complex;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::env;
use std::time::Instant; // Para medir el tiempo
use crc32fast::Hasher; // Para calcular el CRC32

pub fn get_args() -> Result<(String, String), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("Error: Name or size missing".to_string());
    }
    Ok((args[1].clone(), args[2].clone()))
}

/// Calcula si un punto pertenece al conjunto de Mandelbrot
/// Devuelve un valor entre 0 y 255 según el número de iteraciones antes de que escape
fn mandelbrot(c: Complex<f64>, max_iterations: u32) -> u8 {
    let mut z = Complex::new(0.0, 0.0);
    for n in 0..max_iterations {
        if z.norm_sqr() > 4.0 {
            return n as u8;
        }
        z = z * z + c;
    }
    max_iterations as u8
}

fn mandelbrot_set(
    width: usize,
    height: usize,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    max_iterations: u32,
) -> Vec<u8> {
    let mut img = vec![0; width * height];
    let scale_x = (max_x - min_x) / width as f64;
    let scale_y = (max_y - min_y) / height as f64;

    for y in 0..height {
        for x in 0..width {
            let real = min_x + x as f64 * scale_x;
            let imag = min_y + y as f64 * scale_y;
            let c = Complex::new(real, imag);
            img[y * width + x] = mandelbrot(c, max_iterations);
        }
    }

    img
}

fn write_pgm(filename: &str, img: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "P5")?;
    writeln!(writer, "{} {}", width, height)?;
    writeln!(writer, "255")?;
    writer.write_all(img)?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    let (file_name, size) = get_args().expect("Error getting args");
    let size: u32 = size.parse().expect("Error parsing size");
    let dimension = 2_usize.pow(9 + size);

    // Start timing computation
    let start = Instant::now();

    let img = mandelbrot_set(dimension, dimension, -2.0, 0.5, -1.25, 1.25, 255);

    let computation_time = start.elapsed(); // End timing computation

    // Write the PGM file
    write_pgm(&file_name, &img, dimension, dimension)?;

    // Calculate total pixels
    let total_pixels = (dimension * dimension) as f64;

    // Compute megapixels per second
    let time_in_seconds = computation_time.as_secs_f64();
    let mp_per_second = total_pixels / (1_000_000.0 * time_in_seconds);

    // Calculate CRC32 checksum
    let mut hasher = Hasher::new();
    hasher.update(&img);
    let checksum = hasher.finalize();

    // Output results
    
    println!("Checksum: 0x{:X}, computed {:.0} pixels in {:?} = {:.2} Mpx/s", checksum, total_pixels, computation_time, mp_per_second);
    

    Ok(())
}
