//! Build script for naturalneighbor3d
//!
//! This script configures the build environment based on the target platform
//! and available features, similar to how orjson optimizes its build.

use std::env;

fn main() {
    // Rerun build script if these files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=TARGET");

    // Configure optimization features
    configure_optimization_features();

    // Configure Python integration
    configure_python_features();

    // Configure target-specific features
    configure_target_features();
}

fn configure_optimization_features() {
    // Enable SIMD optimizations on supported targets
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    match target_arch.as_str() {
        "x86_64" => {
            println!("cargo:rustc-cfg=feature=\"simd\"");
            println!("cargo:rustc-cfg=feature=\"avx2\"");

            // Check if we can use more advanced SIMD
            if is_feature_available("avx512f") {
                println!("cargo:rustc-cfg=feature=\"avx512\"");
            }
        }
        "aarch64" => {
            println!("cargo:rustc-cfg=feature=\"simd\"");
            println!("cargo:rustc-cfg=feature=\"neon\"");
        }
        _ => {
            // Fallback to scalar implementations
            println!("cargo:rustc-cfg=feature=\"scalar_only\"");
        }
    }

    // Enable fast math optimizations in release mode
    if env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-cfg=feature=\"fast_math\"");
    }
}

fn configure_python_features() {
    // Get Python configuration from pyo3
    let python_config = pyo3_build_config::get();
    if python_config.implementation == pyo3_build_config::PythonImplementation::CPython {
        println!("cargo:rustc-cfg=feature=\"cpython\"");
    } else {
        // We only support CPython for now
        panic!("Only CPython is supported");
    }
}

fn configure_target_features() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();

    // Configure platform-specific optimizations
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-cfg=feature=\"linux\"");

            // Enable memory optimizations on Linux
            println!("cargo:rustc-cfg=feature=\"mmap\"");
        }
        "windows" => {
            println!("cargo:rustc-cfg=feature=\"windows\"");
        }
        "macos" => {
            println!("cargo:rustc-cfg=feature=\"macos\"");
        }
        _ => {}
    }

    // Configure for Unix-like systems
    if target_family == "unix" {
        println!("cargo:rustc-cfg=feature=\"unix\"");
    }

    // Configure threading model
    if env::var("CARGO_CFG_TARGET_HAS_ATOMIC")
        .unwrap_or_default()
        .contains("64")
    {
        println!("cargo:rustc-cfg=feature=\"atomic64\"");
    }
}

fn is_feature_available(feature: &str) -> bool {
    // This is a simplified check - in practice you might use
    // more sophisticated feature detection
    match std::process::Command::new("rustc")
        .args(["+nightly", "--print", "target-features"])
        .output()
    {
        Ok(output) => {
            let features = String::from_utf8_lossy(&output.stdout);
            features.contains(feature)
        }
        Err(_) => false,
    }
}
