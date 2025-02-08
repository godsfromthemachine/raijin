use anyhow::{anyhow, Result};
use ort::{Environment, Session, SessionBuilder, Value, ExecutionProvider};
use ort::execution_providers::CPUExecutionProviderOptions;
use ndarray::{Array, CowArray, IxDyn, Array1};
use std::sync::Arc;
use half::f16;
//use tokenizers::Tokenizer; // Remove tokenizers
use std::fs::File;
use std::io::{BufReader, Read};
use serde_json::{Value as JsonValue, Map};
use serde_json::from_str;
use std::collections::HashMap;

// Constants from Netron (model-fixed-dims.onnx):
const TOTAL_SEQUENCE: usize = 1024;    // for attention_mask and position_ids
// There are 28 pairs of past key/value tensors (indices 0 to 27)
const NUM_LAYERS: usize = 28;
const HEAD_DIM: usize = 128;         // head dimension for past key/values
const PAST_SEQ_LEN: usize = TOTAL_SEQUENCE - 1; // 1023

struct CustomTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    bos_token_id: u32,
}

impl CustomTokenizer {
    fn new(vocab_file: &str) -> Result<Self> {
        let file = File::open(vocab_file)?;
        let reader = BufReader::new(file);
        let json: JsonValue = serde_json::from_reader(reader)?;

        let vocab_map = match json.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
            Some(vocab) => {
                let mut map = HashMap::new();
                let mut reverse_map = HashMap::new();
                 for (token, id) in vocab {
                    if let JsonValue::Number(num) = id {
                        if let Some(id_u32) = num.as_u64().map(|x| x as u32) {
                            map.insert(token.clone(), id_u32);
                            reverse_map.insert(id_u32, token.clone());
                        }
                    }
                }
                (map, reverse_map)
            }
            None => {
                return Err(anyhow!("Vocabulary not found in tokenizer.json"));
            }
        };

        // Extract BOS token ID from added_tokens section of tokenizer.json
        let bos_token_id = json
            .get("added_tokens")
            .and_then(|tokens| tokens.as_array())
            .and_then(|tokens_array| {
                tokens_array.iter().find(|token| {
                    token.get("content").and_then(|c| c.as_str()) == Some("<｜begin▁of▁sentence｜>")
                })
            })
            .and_then(|bos_token| bos_token.get("id"))
            .and_then(|id| id.as_u64())
            .map(|id| id as u32)
            .ok_or(anyhow!("BOS token not found in tokenizer.json"))?;

        Ok(CustomTokenizer {
            vocab: vocab_map.0,
            reverse_vocab: vocab_map.1,
            bos_token_id,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // Very basic tokenization: split by whitespace
        text.split_whitespace()
            .map(|word| {
                *self.vocab.get(word).unwrap_or(&0) // Replace with 0 if not found
            })
            .collect()
    }

    fn decode(&self, token_ids: &[u32]) -> String {
        let mut decoded_string = String::new();
        for &id in token_ids {
            match self.reverse_vocab.get(&id) {
                Some(token) => {
                    decoded_string.push_str(token);
                    decoded_string.push_str(" ");
                }
                None => {
                    decoded_string.push_str("[UNK] "); // Mark unknown tokens
                }
            }
        }
        decoded_string
    }
}

fn main() -> Result<()> {
    // --- Bypass Tokenizer ---
    // Instead of loading a tokenizer ---
    let prompt = "Hello, world!";
    println!("Prompt: '{}'", prompt);

    // --- Load Tokenizer ---
    let tokenizer_path = "models/tokenizer.json"; // Specify the path to the downloaded tokenizer.json

    // Load the CustomTokenizer
    let tokenizer = CustomTokenizer::new(tokenizer_path)
        .map_err(|e| anyhow!("Error loading CustomTokenizer: {}", e))?;

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

    // --- Build Input Tensors ---
    // 1. input_ids: int64[1, 1] (BOS token)
    let input_ids_array = Array::from_elem((1, 1), tokenizer.bos_token_id as i64);
    let input_ids_cow: &'static CowArray<i64, IxDyn> =
        Box::leak(Box::new(CowArray::from(input_ids_array.into_dyn())));
    let input_ids_tensor = Value::from_array(allocator, input_ids_cow)?;

    // 2. attention_mask: int64[1, TOTAL_SEQUENCE]
    let mut attention_mask_array = Array::<i64, _>::zeros((1, TOTAL_SEQUENCE));
    attention_mask_array[[0, 0]] = 1; // Only attend to the BOS token
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
        Some(idx) => {
            println!("Token id: {}", idx);
            idx as u32
        }
        None => {
            eprintln!("Argmax failed, array is empty?");
            return Err(anyhow!("Argmax failed, array is empty?"));
        }
    };

   // 3. Decode the token ID using the tokenizer
    let decoded_text = tokenizer.decode(&[token_id].to_vec());

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