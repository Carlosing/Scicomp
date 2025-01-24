//Carlos Alberto Escobedo Lopez

use num::Complex;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::env;
use std::time::Instant; // For measuring time
use crc32fast::Hasher; // For calculating CRC32
use rayon::prelude::*; // For parallelization
use sys_info; // For obtaining system information
use num_cpus; // For obtaining the number of cores

/// Parses command line arguments and returns the file name and size.
/// Returns an error if the number of arguments is incorrect.
pub fn get_args() -> Result<(String, String), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("Error: Name or size missing".to_string());
    }
    Ok((args[1].clone(), args[2].clone()))
}

/// Determines if a point belongs to the Mandelbrot set.
/// Returns a value between 0 and 255 based on the number of iterations before it escapes.
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

/// Generates the Mandelbrot set for a given image size and coordinate range.
/// Uses parallel processing to speed up the computation.
fn mandelbrot_set(
    width: usize,
    height: usize,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    max_iterations: u32,
) -> Vec<u8> {
    let scale_x = (max_x - min_x) / width as f64;
    let scale_y = (max_y - min_y) / height as f64;

    // Create a vector to store the image
    let mut img = vec![0; width * height];

    //The original sequential code which will be later parallelized
    // for y in 0..height {
    //     for x in 0..width {
    //         let real = min_x + x as f64 * scale_x;
    //         let imag = min_y + y as f64 * scale_y;
    //         let c = Complex::new(real, imag);
    //         img[y * width + x] = mandelbrot(c, max_iterations);
    //     }
    // }



    // Parallelize the Mandelbrot computation using parallel iterators
    img.par_chunks_mut(width) // Divide the image into rows for parallel processing
        .enumerate()          // Include the row index (y)
        .for_each(|(y, row)| {
            let imag = min_y + y as f64 * scale_y;
            for x in 0..width {
                let real = min_x + x as f64 * scale_x;
                let c = Complex::new(real, imag);
                row[x] = mandelbrot(c, max_iterations);
            }
        });


    img
}

/// Writes the image data to a PGM file.
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

    println!("Checksum: 0x{:X}, computed {:.0} pixels in {:?} = {:.2} Mpx/s", checksum, total_pixels, computation_time, mp_per_second);

    // Gather system information
    let cpu_count = sys_info::cpu_num().unwrap_or(0); // Number of CPUs
    let os = sys_info::os_type().unwrap_or_else(|_| "Unknown".to_string());
    let compiler = env::var("RUSTC_VERSION").unwrap_or_else(|_| "rustc".to_string());
    let cores = num_cpus::get_physical();
    let threads = num_cpus::get();

    // Output results in a detailed table
    println!("+----------------+------------------+------------+-------+------------+----------+---------+------+-------+");
    println!("| CPU            | OS               | Compiler   | Cores | Threads    | Scale    | S-Thread| MPI  | Note  |");
    println!("+----------------+------------------+------------+-------+------------+----------+---------+------+-------+");
    println!("| {:<14} | {:<16} | {:<10} | {:<5} | {:<10} | {:<8} | {:<7} | {:<4} | {:<5} |",
        format!("{} CPUs", cpu_count), os, compiler, cores, threads, format!("{}x{}", dimension, dimension), "Rayon", "N/A", "");
    println!("+----------------+------------------+------------+-------+------------+----------+---------+------+-------+");

    Ok(())
}
