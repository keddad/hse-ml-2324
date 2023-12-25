use anyhow::Result;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyRuntimeError;
use std::time::Instant;
use tract_ndarray::Array;
use tract_onnx::prelude::*;

#[pyfunction]
fn run_model() -> PyResult<(String, f64)> {
    run_model_internal().map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("Error: {}", e)))
}

fn run_model_internal() -> Result<(String, f64)> {
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(0, f32::fact([1, 3, 224, 224]).into())?
        .into_optimized()?
        .into_runnable()?;

    let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.0, 0.0, 0.0])?;
    let std = Array::from_shape_vec((1, 3, 1, 1), vec![1.0, 1.0, 1.0])?;

    let img = image::open(image_path)?.to_rgb8();
    let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = ((tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32 / 224.0
    }) - mean)).into();

    // run model with speed measurement
    let start = Instant::now();
    let result = model.run(tvec!(image.into()))?;
    let duration = start.elapsed();

    // Construct the 'hours:minutes' string
    let time_string = &result;

    Ok((time_string, duration))
}

#[pymodule]
fn onnx_inference(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_model, m)?)?;
    Ok(())
}
