use anyhow::{anyhow, Result};
use ort::{Environment, Session, SessionBuilder, Value, ExecutionProvider};
use ort::execution_providers::CPUExecutionProviderOptions;
use ndarray::{Array, CowArray, IxDyn, Array1, s, Array4};
use std::sync::Arc;
use half::f16;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

// Constants matching model requirements
const TOTAL_SEQUENCE: usize = 1024; 
const NUM_LAYERS: usize = 28;
const HEAD_DIM: usize = 128;
const PAST_SEQ_LEN: usize = TOTAL_SEQUENCE - 1;
const MAX_GENERATION_LENGTH: usize = 256;

// Sampling configuration
const TEMPERATURE: f32 = 0.7;
const TOP_K: usize = 50;             
const TOP_P: f32 = 0.9;              

fn main() -> Result<()> {
    // --- Load Tokenizer ---
    let tokenizer_path = "models/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("Error loading tokenizer: {}", e))?;

    // Apply chat template
    let chat_text = format!("<｜begin▁of▁sentence｜><｜User｜>Hello, world!<｜Assistant｜>");
    println!("Chat text: '{}'", chat_text);

    // --- Encode the Prompt ---
    let encoding = tokenizer.encode(chat_text, false)
        .map_err(|e| anyhow!("Error encoding prompt: {}", e))?;
    let encoded_prompt: Vec<u32> = encoding.get_ids().to_vec();
    println!("Encoded prompt: {:?}", encoded_prompt);
    let num_prompt_tokens = encoded_prompt.len();

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
    
    let session: Session = session_builder.with_model_from_file("models/model-fixed-dims.onnx")?;
    let allocator = session.allocator();

    // --- Initialize Input Tensors ---
    // Start with the first token
    let first_token = encoded_prompt[0];
    let mut input_ids_array = Array::from_elem((1, 1), first_token as i64);

    // Initialize attention_mask_array
    let mut attention_mask_array = Array::<i64, _>::zeros((1, TOTAL_SEQUENCE));
    attention_mask_array[[0, 0]] = 1;

    // Initialize position_ids_array with incremental positions
    let position_ids_array = Array::from_shape_fn((1, TOTAL_SEQUENCE), |(_, j)| j as i64);

    // Initialize tree attention array for causal attention
    let mut tree_att = Array::<f16, _>::from_elem((TOTAL_SEQUENCE, TOTAL_SEQUENCE), f16::from_f32(-65504.0));
    for i in 0..TOTAL_SEQUENCE {
        for j in 0..=i {
            tree_att[[i, j]] = f16::from_f32(0.0);
        }
    }
    let tree_attention_array = tree_att.into_shape((1, 1, TOTAL_SEQUENCE, TOTAL_SEQUENCE))?;

    // --- Initialize Past Key/Values ---
    let past_kv_shape = [1, 2, PAST_SEQ_LEN, HEAD_DIM];
    let mut past_key_values_vec: Vec<Array4<f16>> = Vec::new();
    for _ in 0..NUM_LAYERS {
        past_key_values_vec.push(Array4::from_elem(past_kv_shape, f16::from_f32(0.0)));
    }

    // --- Generation Loop ---
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut current_length: usize = 0;
    let mut past_sequence_length = 1;
    let mut prompt_index = 1;
    let mut rng = rand::thread_rng();

    while current_length < MAX_GENERATION_LENGTH {
        // --- Create Input Tensors ---
        let input_ids_cow: &'static CowArray<i64, IxDyn> =
            Box::leak(Box::new(CowArray::from(input_ids_array.clone().into_dyn())));
        let input_ids_tensor = Value::from_array(allocator, input_ids_cow)?;

        let attention_mask_cow: &'static CowArray<i64, IxDyn> =
            Box::leak(Box::new(CowArray::from(attention_mask_array.clone().into_dyn())));
        let attention_mask_tensor = Value::from_array(allocator, attention_mask_cow)?;

        let position_ids_cow: &'static CowArray<i64, IxDyn> =
            Box::leak(Box::new(CowArray::from(position_ids_array.clone().into_dyn())));
        let position_ids_tensor = Value::from_array(allocator, position_ids_cow)?;

        let tree_attention_cow: &'static CowArray<f16, IxDyn> =
            Box::leak(Box::new(CowArray::from(tree_attention_array.clone().into_dyn())));
        let tree_attention_tensor = Value::from_array(allocator, tree_attention_cow)?;

        // Create past key/value tensors
        let mut inputs: Vec<Value> = vec![
            input_ids_tensor,
            attention_mask_tensor,
            position_ids_tensor,
            tree_attention_tensor,
        ];

        for layer in &past_key_values_vec {
            let past_kv_cow: &'static CowArray<f16, IxDyn> =
                Box::leak(Box::new(CowArray::from(layer.clone().into_dyn())));
            let key_tensor = Value::from_array(allocator, past_kv_cow)?;
            let value_tensor = Value::from_array(allocator, past_kv_cow)?;
            inputs.push(key_tensor);
            inputs.push(value_tensor);
        }

        // --- Run Inference ---
        let outputs = session.run(inputs)?;

        // --- Process Output ---
        let logits_value = &outputs[0];
        let logits_data = match logits_value.try_extract::<f16>() {
            Ok(tensor) => tensor.view().to_owned(),
            Err(e) => return Err(anyhow!("Failed to extract tensor: {}", e)),
        };

        let logits_shape = logits_data.shape();
        let num_tokens = logits_shape[logits_shape.len() - 1];
        let logits_reshaped = logits_data.into_shape(num_tokens)?;

        // Convert f16 logits to f32 and apply temperature
        let logits_f32: Array1<f32> = logits_reshaped.map(|x| x.to_f32() / TEMPERATURE);
        
        // Apply softmax and sampling
        let max_logit = logits_f32.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Array1<f32> = logits_f32.map(|&x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let mut probs = exp_logits / sum_exp;

        // Apply top-k filtering
        let mut top_k_indices: Vec<_> = (0..probs.len()).collect();
        top_k_indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());
        top_k_indices.truncate(TOP_K);
        
        let mut top_k_probs = Array1::zeros(probs.len());
        for &idx in &top_k_indices {
            top_k_probs[idx] = probs[idx];
        }
        probs = top_k_probs;

        // Apply nucleus sampling
        let mut cumsum = 0.0;
        let mut nucleus_probs = Array1::zeros(probs.len());
        for &idx in &top_k_indices {
            cumsum += probs[idx];
            nucleus_probs[idx] = probs[idx];
            if cumsum >= TOP_P {
                break;
            }
        }

        let sum_nucleus = nucleus_probs.sum();
        if sum_nucleus > 0.0 {
            nucleus_probs.mapv_inplace(|x| x / sum_nucleus);
        }

        // Sample token
        let dist = WeightedIndex::new(nucleus_probs.iter())?;
        let token_id = dist.sample(&mut rng) as u32;

        // Check for EOS token
        if token_id == tokenizer.token_to_id("<｜end▁of▁sentence｜>").unwrap_or(0) {
            println!("\nEOS Token Generated");
            break;
        }

        // Update generated tokens
        generated_tokens.push(token_id);
        current_length += 1;

        // Decode and print token
        let decoded_token = match tokenizer.decode(&[token_id], true) {
            Ok(token) => token,
            Err(_) => "[UNK]".to_string(),
        };
        print!("{}", decoded_token);
        io::stdout().flush()?;

        // Update input for next iteration
        if prompt_index < num_prompt_tokens {
            input_ids_array = Array::from_elem((1, 1), encoded_prompt[prompt_index] as i64);
            prompt_index += 1;
        } else {
            input_ids_array = Array::from_elem((1, 1), token_id as i64);
        }

        // Update attention mask
        if past_sequence_length < TOTAL_SEQUENCE {
            attention_mask_array[[0, past_sequence_length]] = 1;
            past_sequence_length += 1;
        }

        // Update past key/values
        for i in 0..NUM_LAYERS {
            let key_output = &outputs[1 + i * 2];
            let key_data = match key_output.try_extract::<f16>() {
                Ok(tensor) => tensor.view().to_owned(),
                Err(e) => return Err(anyhow!("Failed to extract key tensor: {}", e)),
            };

            let mut new_past_kv = Array4::from_elem(past_kv_shape, f16::from_f32(0.0));
            new_past_kv.slice_mut(s![.., .., ..past_sequence_length-1, ..])
                .assign(&key_data.slice(s![.., .., ..past_sequence_length-1, ..]));
            past_key_values_vec[i] = new_past_kv;
        }
    }

    println!("\nInference complete.");
    let decoded_text = tokenizer.decode(&generated_tokens, true)
        .map_err(|e| anyhow!("Failed to decode text: {}", e))?;
    println!("Full generated text: {}", decoded_text);

    Ok(())
}