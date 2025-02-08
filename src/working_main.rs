use anyhow::Result;
use ort::{Environment, SessionBuilder, Value, ExecutionProvider};
use ort::execution_providers::CPUExecutionProviderOptions;
use ndarray::{Array, CowArray, IxDyn};
use std::sync::Arc;
use half::f16;

// Constants from Netron (model-fixed-dims.onnx):
const TOTAL_SEQUENCE: usize = 1024;   // for attention_mask and position_ids
// There are 28 pairs of past key/value tensors (indices 0 to 27)
const NUM_LAYERS: usize = 28;
const HEAD_DIM: usize = 128;          // head dimension for past key/values
const PAST_SEQ_LEN: usize = TOTAL_SEQUENCE - 1; // 1023

fn main() -> Result<()> {
    // --- Bypass Tokenizer ---
    // Instead of loading a tokenizer, we use a constant BOS token (set to 1).
    let bos_id: i64 = 1;
    let prompt = "Hello, world!";
    println!("Prompt: '{}' with BOS token id: {}", prompt, bos_id);

    // --- Create ONNX Runtime Environment and Session ---
    let environment = Arc::new(
        Environment::builder()
            .with_name("DeepSeek")
            .build()?
    );
    let mut session_builder = SessionBuilder::new(&environment)?;
    session_builder = session_builder.with_intra_threads(1)?;
    session_builder = session_builder.with_execution_providers(vec![
        ExecutionProvider::CPU(CPUExecutionProviderOptions::default())
    ])?;
    // Use the fixed-dims model.
    let session = session_builder.with_model_from_file("models/model-fixed-dims.onnx")?;
    let allocator = session.allocator();

    // --- Build Input Tensors ---
    // We leak the CowArrays so that they live for the entire program.
    
    // 1. input_ids: int64[1,1]
    let input_ids_array = Array::<i64, _>::from_elem((1, 1), bos_id);
    let input_ids_cow: &'static CowArray<i64, IxDyn> =
        Box::leak(Box::new(CowArray::from(input_ids_array.into_dyn())));
    let input_ids_tensor = Value::from_array(allocator, input_ids_cow)?;

    // 2. attention_mask: int64[1,1024] (only the first token active)
    let mut attention_mask_array = Array::<i64, _>::zeros((1, TOTAL_SEQUENCE));
    attention_mask_array[[0, 0]] = 1;
    let attention_mask_cow: &'static CowArray<i64, IxDyn> =
        Box::leak(Box::new(CowArray::from(attention_mask_array.into_dyn())));
    let attention_mask_tensor = Value::from_array(allocator, attention_mask_cow)?;

    // 3. position_ids: int64[1,1024] with sequential indices 0..1023
    let position_ids_array = Array::from_shape_fn((1, TOTAL_SEQUENCE), |(_, j)| j as i64);
    let position_ids_cow: &'static CowArray<i64, IxDyn> =
        Box::leak(Box::new(CowArray::from(position_ids_array.into_dyn())));
    let position_ids_tensor = Value::from_array(allocator, position_ids_cow)?;

    // 4. tree_attention: float16[1,1,1024,1024]
    let mut tree_att = Array::<f16, _>::from_elem((TOTAL_SEQUENCE, TOTAL_SEQUENCE), f16::from_f32(-65504.0));
    for i in 0..TOTAL_SEQUENCE {
        for j in 0..=i {
            tree_att[[i, j]] = f16::from_f32(0.0);
        }
    }
    let tree_attention_array = tree_att.into_shape((1, 1, TOTAL_SEQUENCE, TOTAL_SEQUENCE))?;
    let tree_attention_cow: &'static CowArray<f16, IxDyn> =
        Box::leak(Box::new(CowArray::from(tree_attention_array.into_dyn())));
    let tree_attention_tensor = Value::from_array(allocator, tree_attention_cow)?;

    // 5. Past key/value tensors: for each of NUM_LAYERS layers,
    //    create two tensors (key and value) each of shape float16[1,2,1023,128].
    let past_kv_array = Array::<f32, _>::zeros((1, 2, PAST_SEQ_LEN, HEAD_DIM));
    let past_kv_array_f16 = past_kv_array.map(|&x| f16::from_f32(x));
    let past_kv_cow: &'static CowArray<f16, IxDyn> =
        Box::leak(Box::new(CowArray::from(past_kv_array_f16.into_dyn())));
    let mut past_key_values: Vec<Value> = Vec::new();
    for _ in 0..NUM_LAYERS {
        let key_tensor = Value::from_array(allocator, past_kv_cow)?;
        let value_tensor = Value::from_array(allocator, past_kv_cow)?;
        past_key_values.push(key_tensor);
        past_key_values.push(value_tensor);
    }

    // --- Assemble the Full Input Vector ---
    // The expected order is:
    // 1. input_ids
    // 2. attention_mask
    // 3. position_ids
    // 4. tree_attention
    // 5. All past key/value tensors in order.
    let mut inputs: Vec<Value> = Vec::new();
    inputs.push(input_ids_tensor);
    inputs.push(attention_mask_tensor);
    inputs.push(position_ids_tensor);
    inputs.push(tree_attention_tensor);
    inputs.extend(past_key_values);

    // --- Run Inference ---
    let outputs = session.run(inputs)?;

    // Print the outputs.
    println!("Inference outputs:");
    for (i, output) in outputs.iter().enumerate() {
        println!("Output {}: {:?}", i, output);
    }

    Ok(())
}
