use std::time::Instant;
use tract_ndarray::Array;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path("model.pth")?
        // specify input type and shape
        .with_input_fact(0, f32::fact([1, 3, 224, 244]).into())?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    // Imagenet mean and standard deviation
    let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.0, 0.0, 0.0])?;
    let std = Array::from_shape_vec((1, 3, 1, 1), vec![1.0, 1.0, 1.0])?;

    let img = image::open("example.png").unwrap().to_rgb8();
    let resized = image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor =
        ((tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
            resized[(x as _, y as _)][c] as f32 / 224.0
        }) - mean))
            .into();

    // run model with speed measurement
    let start = Instant::now()
    let result = model.run(tvec!(image.into()))?;
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {best:?}");

    Ok(())
}