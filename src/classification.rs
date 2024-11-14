use image::ImageReader;
use image::{DynamicImage, RgbImage};
use ndarray::{Array, Dim};
use std::fs;
use std::io::BufRead;
use std::error::Error;
use std::sync::Arc;

use crate::wasi::nn::{
    graph::{Graph, GraphBuilder, load, ExecutionTarget, GraphEncoding},
    tensor::{Tensor, TensorData, TensorDimensions, TensorType},
};

pub struct Classifier {
    graph: Arc<Graph>,
    labels: Vec<String>,
}

impl Classifier {
    pub fn new(model_path: &str, labels_path: &str) -> Result<Self, Box<dyn Error>> {
        let model: GraphBuilder = fs::read(model_path)?;
        let graph = load(&[model], GraphEncoding::Onnx, ExecutionTarget::Cpu)?;
        let labels = fs::read(labels_path)?;
        let class_labels: Vec<String> = labels.lines().map(|line| line.unwrap()).collect();

        Ok(Self {
            graph: Arc::new(graph),
            labels: class_labels,
        })
    }

    pub fn classify_image(&self, image_path: &str) -> Result<Vec<(String, f32)>, Box<dyn Error>> {
        let exec_context = Graph::init_execution_context(&self.graph)?;

        // Prepare input tensor
        let dimensions: TensorDimensions = vec![1, 3, 224, 224];
        let data: TensorData = image_to_tensor(image_path.to_string(), 224, 224);
        let tensor = Tensor::new(&dimensions, TensorType::Fp32, &data);
        exec_context.set_input("data", tensor)?;

        // Execute inference
        exec_context.compute()?;

        // Get output and process
        let output_data = exec_context.get_output("squeezenet0_flatten0_reshape0")?.data();
        let output_f32 = bytes_to_f32_vec(output_data);

        let output_shape = [1, 1000, 1, 1];
        let output_tensor = Array::from_shape_vec(output_shape, output_f32)?;

        // Compute softmax
        let exp_output = output_tensor.mapv(|x| x.exp());
        let sum_exp_output = exp_output.sum_axis(ndarray::Axis(1));
        let softmax_output = exp_output / &sum_exp_output;

        // Get top 3 predictions
        let mut sorted = softmax_output
            .index_axis(ndarray::Axis(0), 0)
            .iter()
            .enumerate()
            .collect::<Vec<(usize, &f32)>>();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_classes = sorted
            .into_iter()
            .rev()
            .take(3)
            .map(|(index, &probability)| (self.labels[index].clone(), probability))
            .collect();

        Ok(top_classes)
    }
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    data.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let pixels = ImageReader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
    let bgr_img: RgbImage = dyn_img.to_rgb8();

    // Get pixel data
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];

    // Prepare the tensor data
    let mut u8_f32_arr: Vec<u8> = Vec::with_capacity(raw_u8_arr.len() * 4);

    // Normalizing values
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    for (i, &pixel) in raw_u8_arr.iter().enumerate() {
        let u8_f32: f32 = pixel as f32;
        let rgb_iter = i % 3;

        let norm_u8_f32: f32 = (u8_f32 / 255.0 - mean[rgb_iter]) / std[rgb_iter];
        u8_f32_arr.extend_from_slice(&norm_u8_f32.to_ne_bytes());
    }

    u8_f32_arr
}
