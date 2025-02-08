use anyhow::{anyhow, Result};
use ort::{Environment, Session, SessionBuilder, Value, ExecutionProvider};
use ort::execution_providers::CPUExecutionProviderOptions;
use ndarray::{Array, CowArray, IxDyn, Array1};
use std::sync::Arc;
use half::f16;
use tokenizers::Tokenizer; // Import Tokenizer
use std::fs::File;
use std::io::{BufReader, Read};
use serde_json::Value as JsonValue;
use serde_json::from_str;

// Constants from Netron (model-fixed-dims.onnx):
const TOTAL_SEQUENCE: usize = 1024;    // for attention_mask and position_ids
// There are 28 pairs of past key/value tensors (indices 0 to 27)
const NUM_LAYERS: usize = 28;
const HEAD_DIM: usize = 128;         // head dimension for past key/values
const PAST_SEQ_LEN: usize = TOTAL_SEQUENCE - 1; // 1023

fn main() -> Result<()> {
    // --- Bypass Tokenizer ---
    // Instead of loading a tokenizer ---
    let bos_id: i64 = 1;
    let prompt = "Hello, world!";
    println!("Prompt: '{}' with BOS token id: {}", prompt, bos_id);

    // --- Load Tokenizer ---
    let tokenizer_name = "Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx"; // Specify the model name
    let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None).map_err(|e| anyhow!("Error loading tokenizer from pretrained: {}", e))?;

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
    let session: Session = session_builder.with_model_from_file("models/model-fixed-dims.onnx")?;
    let allocator = session.allocator();

    // --- Tokenize the Prompt ---
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow!("Error encoding prompt: {:?}", e))?;
    let token_ids = encoding.get_ids();
    let sequence_length = token_ids.len();

    // --- Build Input Tensors ---
    // 1. input_ids: int64[1, TOTAL_SEQUENCE] (pad to TOTAL_SEQUENCE)
    let mut input_ids_array = Array::<i64, _>::zeros((1, TOTAL_SEQUENCE));
    for i in 0..sequence_length {
        input_ids_array[[0, i]] = token_ids[i] as i64;
    }
    let input_ids_cow: &'static CowArray<i64, IxDyn> =
        Box::leak(Box::new(CowArray::from(input_ids_array.into_dyn())));
    let input_ids_tensor = Value::from_array(allocator, input_ids_cow)?;

    // 2. attention_mask: int64[1, TOTAL_SEQUENCE]
    let mut attention_mask_array = Array::<i64, _>::zeros((1, TOTAL_SEQUENCE));
    for i in 0..sequence_length {
        attention_mask_array[[0, i]] = 1;
    }
    let attention_mask_cow: &'static CowArray<i64, IxDyn> =
        Box::leak(Box::new(CowArray::from(attention_mask_array.into_dyn())));
    let attention_mask_tensor = Value::from_array(allocator, attention_mask_cow)?;

    // 3. position_ids: int64[1, TOTAL_SEQUENCE]
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

    // --- Process Output ---
    // 1. Extract logits
    let logits_value = &outputs[0];

    // Extract data as f16
    let logits_data = match logits_value.try_extract::<f16>() {
        Ok(tensor) => {
            let view = tensor.view();
            view.to_owned()
        }
        Err(e) => return Err(anyhow!("Failed to extract tensor: {}", e)),
    };

    // Get shape
    let logits_shape = logits_data.shape();
    let num_tokens = logits_shape[logits_shape.len() - 1];

    // Reshape the array
    let logits_reshaped = logits_data.into_shape(num_tokens)?;

    // Find argmax
    let argmax_result = argmax(&logits_reshaped);

    let token_id = match argmax_result {
        Some(idx) => idx as u32,
        None => {
            eprintln!("Argmax failed, array is empty?");
            return Err(anyhow!("Argmax failed, array is empty?"));
        }
    };

    // 3. Decode the token ID using the tokenizer
    let decoded_text = tokenizer.decode(&[token_id], false)
        .map_err(|e| anyhow!("Error decoding token: {:?}", e))?;

    // Print the outputs.
    println!("Inference outputs:");
    println!("Decoded text: {}", decoded_text);

    Ok(())
}

fn get_json_type(value: &JsonValue) -> &'static str {
    match value {
        JsonValue::Null => "Null",
        JsonValue::Bool(_) => "Bool",
        JsonValue::Number(_) => "Number",
        JsonValue::String(_) => "String",
        JsonValue::Array(_) => "Array",
        JsonValue::Object(_) => "Object",
    }
}

// Helper function to find argmax
fn argmax<T: PartialOrd>(array: &Array1<T>) -> Option<usize> {
    array
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
}