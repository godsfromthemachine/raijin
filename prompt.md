I'm writing a Rust application to run CPU only inference in ONNX for https://huggingface.co/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx


I have this code that works without any compile time and runtime errors


`src/main.rs`

```rust
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
const TOTAL_SEQUENCE: usize = 1024;   // Changed from 128 to match model expectations
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
```



here's the Cargo.toml


```toml
[package]
name = "deepseek-inference"
version = "0.1.0"
edition = "2021"

[dependencies]
ort = "1.16.3"
anyhow = "1.0"
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8"
half = "2.4.1"
tokenizers = "0.21"
```


when I run it, the output is strange, I want readable text, but it starts generating some gibberish


```powershell
 cargo run --bin deepseek-inference
   Compiling deepseek-inference v0.1.0 (C:\Development\godsfromthemachine\raijin)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.01s
     Running `target\debug\deepseek-inference.exe`
Chat text: '<｜begin▁of▁sentence｜><｜User｜>Hello, world!<｜Assistant｜>'
Encoded prompt: [151646, 151644, 9707, 11, 1879, 0, 151645]
 ensure0其als.或者其他12
```

here's how the model's files look like

```bash
$ ls -a models/

.

..

.git

.gitattributes

EtGlowExecutionProvider_GLOW_graph_Extracted_from_-Extracted_from_-Extracted_from_-Extracted_from_-main_graph----_2858289663424555064_0_0_0.onnx

EtGlowExecutionProvider_GLOW_graph_Extracted_from_-Extracted_from_-Extracted_from_-Extracted_from_-main_graph----_9203402667311998790_0_0_0.onnx

README.md

model-fixed-dims.onnx

model.embed_tokens.weight

model.layers.0.input_layernorm.weight

model.layers.0.post_attention_layernorm.weight

model.layers.0.self_attn.q_proj.bias

model.layers.1.input_layernorm.weight

model.layers.1.post_attention_layernorm.weight

model.layers.1.self_attn.q_proj.bias

model.layers.10.input_layernorm.weight

model.layers.10.post_attention_layernorm.weight

model.layers.10.self_attn.q_proj.bias

model.layers.11.input_layernorm.weight

model.layers.11.post_attention_layernorm.weight

model.layers.11.self_attn.q_proj.bias

model.layers.12.input_layernorm.weight

model.layers.12.post_attention_layernorm.weight

model.layers.12.self_attn.q_proj.bias

model.layers.13.input_layernorm.weight

model.layers.13.post_attention_layernorm.weight

model.layers.13.self_attn.q_proj.bias

model.layers.14.input_layernorm.weight

model.layers.14.post_attention_layernorm.weight

model.layers.14.self_attn.q_proj.bias

model.layers.15.input_layernorm.weight

model.layers.15.post_attention_layernorm.weight

model.layers.15.self_attn.q_proj.bias

model.layers.16.input_layernorm.weight

model.layers.16.post_attention_layernorm.weight

model.layers.16.self_attn.q_proj.bias

model.layers.17.input_layernorm.weight

model.layers.17.post_attention_layernorm.weight

model.layers.17.self_attn.q_proj.bias

model.layers.18.input_layernorm.weight

model.layers.18.post_attention_layernorm.weight

model.layers.18.self_attn.q_proj.bias

model.layers.19.input_layernorm.weight

model.layers.19.post_attention_layernorm.weight

model.layers.19.self_attn.q_proj.bias

model.layers.2.input_layernorm.weight

model.layers.2.post_attention_layernorm.weight

model.layers.2.self_attn.q_proj.bias

model.layers.20.input_layernorm.weight

model.layers.20.post_attention_layernorm.weight

model.layers.20.self_attn.q_proj.bias

model.layers.21.input_layernorm.weight

model.layers.21.post_attention_layernorm.weight

model.layers.21.self_attn.q_proj.bias

model.layers.22.input_layernorm.weight

model.layers.22.post_attention_layernorm.weight

model.layers.22.self_attn.q_proj.bias

model.layers.23.input_layernorm.weight

model.layers.23.post_attention_layernorm.weight

model.layers.23.self_attn.q_proj.bias

model.layers.24.input_layernorm.weight

model.layers.24.post_attention_layernorm.weight

model.layers.24.self_attn.q_proj.bias

model.layers.25.input_layernorm.weight

model.layers.25.post_attention_layernorm.weight

model.layers.25.self_attn.q_proj.bias

model.layers.26.input_layernorm.weight

model.layers.26.post_attention_layernorm.weight

model.layers.26.self_attn.q_proj.bias

model.layers.27.input_layernorm.weight

model.layers.27.post_attention_layernorm.weight

model.layers.27.self_attn.q_proj.bias

model.layers.3.input_layernorm.weight

model.layers.3.post_attention_layernorm.weight

model.layers.3.self_attn.q_proj.bias

model.layers.4.input_layernorm.weight

model.layers.4.post_attention_layernorm.weight

model.layers.4.self_attn.q_proj.bias

model.layers.5.input_layernorm.weight

model.layers.5.post_attention_layernorm.weight

model.layers.5.self_attn.q_proj.bias

model.layers.6.input_layernorm.weight

model.layers.6.post_attention_layernorm.weight

model.layers.6.self_attn.q_proj.bias

model.layers.7.input_layernorm.weight

model.layers.7.post_attention_layernorm.weight

model.layers.7.self_attn.q_proj.bias

model.layers.8.input_layernorm.weight

model.layers.8.post_attention_layernorm.weight

model.layers.8.self_attn.q_proj.bias

model.layers.9.input_layernorm.weight

model.layers.9.post_attention_layernorm.weight

model.layers.9.self_attn.q_proj.bias

model.norm.weight

model.onnx

onnx__MatMul_8851

onnx__MatMul_8852

onnx__MatMul_8853

onnx__MatMul_8878

onnx__MatMul_8879

onnx__MatMul_8880

onnx__MatMul_8881

onnx__MatMul_8882

onnx__MatMul_8883

onnx__MatMul_8884

onnx__MatMul_8909

onnx__MatMul_8910

onnx__MatMul_8911

onnx__MatMul_8912

onnx__MatMul_8913

onnx__MatMul_8914

onnx__MatMul_8915

onnx__MatMul_8940

onnx__MatMul_8941

onnx__MatMul_8942

onnx__MatMul_8943

onnx__MatMul_8944

onnx__MatMul_8945

onnx__MatMul_8946

onnx__MatMul_8971

onnx__MatMul_8972

onnx__MatMul_8973

onnx__MatMul_8974

onnx__MatMul_8975

onnx__MatMul_8976

onnx__MatMul_8977

onnx__MatMul_9002

onnx__MatMul_9003

onnx__MatMul_9004

onnx__MatMul_9005

onnx__MatMul_9006

onnx__MatMul_9007

onnx__MatMul_9008

onnx__MatMul_9033

onnx__MatMul_9034

onnx__MatMul_9035

onnx__MatMul_9036

onnx__MatMul_9037

onnx__MatMul_9038

onnx__MatMul_9039

onnx__MatMul_9064

onnx__MatMul_9065

onnx__MatMul_9066

onnx__MatMul_9067

onnx__MatMul_9068

onnx__MatMul_9069

onnx__MatMul_9070

onnx__MatMul_9095

onnx__MatMul_9096

onnx__MatMul_9097

onnx__MatMul_9098

onnx__MatMul_9099

onnx__MatMul_9100

onnx__MatMul_9101

onnx__MatMul_9126

onnx__MatMul_9127

onnx__MatMul_9128

onnx__MatMul_9129

onnx__MatMul_9130

onnx__MatMul_9131

onnx__MatMul_9132

onnx__MatMul_9157

onnx__MatMul_9158

onnx__MatMul_9159

onnx__MatMul_9160

onnx__MatMul_9161

onnx__MatMul_9162

onnx__MatMul_9163

onnx__MatMul_9188

onnx__MatMul_9189

onnx__MatMul_9190

onnx__MatMul_9191

onnx__MatMul_9192

onnx__MatMul_9193

onnx__MatMul_9194

onnx__MatMul_9219

onnx__MatMul_9220

onnx__MatMul_9221

onnx__MatMul_9222

onnx__MatMul_9223

onnx__MatMul_9224

onnx__MatMul_9225

onnx__MatMul_9250

onnx__MatMul_9251

onnx__MatMul_9252

onnx__MatMul_9253

onnx__MatMul_9254

onnx__MatMul_9255

onnx__MatMul_9256

onnx__MatMul_9281

onnx__MatMul_9282

onnx__MatMul_9283

onnx__MatMul_9284

onnx__MatMul_9285

onnx__MatMul_9286

onnx__MatMul_9287

onnx__MatMul_9312

onnx__MatMul_9313

onnx__MatMul_9314

onnx__MatMul_9315

onnx__MatMul_9316

onnx__MatMul_9317

onnx__MatMul_9318

onnx__MatMul_9343

onnx__MatMul_9344

onnx__MatMul_9345

onnx__MatMul_9346

onnx__MatMul_9347

onnx__MatMul_9348

onnx__MatMul_9349

onnx__MatMul_9374

onnx__MatMul_9375

onnx__MatMul_9376

onnx__MatMul_9377

onnx__MatMul_9378

onnx__MatMul_9379

onnx__MatMul_9380

onnx__MatMul_9405

onnx__MatMul_9406

onnx__MatMul_9407

onnx__MatMul_9408

onnx__MatMul_9409

onnx__MatMul_9410

onnx__MatMul_9411

onnx__MatMul_9436

onnx__MatMul_9437

onnx__MatMul_9438

onnx__MatMul_9439

onnx__MatMul_9440

onnx__MatMul_9441

onnx__MatMul_9442

onnx__MatMul_9467

onnx__MatMul_9468

onnx__MatMul_9469

onnx__MatMul_9470

onnx__MatMul_9471

onnx__MatMul_9472

onnx__MatMul_9473

onnx__MatMul_9498

onnx__MatMul_9499

onnx__MatMul_9500

onnx__MatMul_9501

onnx__MatMul_9502

onnx__MatMul_9503

onnx__MatMul_9504

onnx__MatMul_9529

onnx__MatMul_9530

onnx__MatMul_9531

onnx__MatMul_9532

onnx__MatMul_9533

onnx__MatMul_9534

onnx__MatMul_9535

onnx__MatMul_9560

onnx__MatMul_9561

onnx__MatMul_9562

onnx__MatMul_9563

onnx__MatMul_9564

onnx__MatMul_9565

onnx__MatMul_9566

onnx__MatMul_9591

onnx__MatMul_9592

onnx__MatMul_9593

onnx__MatMul_9594

onnx__MatMul_9595

onnx__MatMul_9596

onnx__MatMul_9597

onnx__MatMul_9622

onnx__MatMul_9623

onnx__MatMul_9624

onnx__MatMul_9625

onnx__MatMul_9626

onnx__MatMul_9627

onnx__MatMul_9628

onnx__MatMul_9653

onnx__MatMul_9654

onnx__MatMul_9655

onnx__MatMul_9656

onnx__MatMul_9657

onnx__MatMul_9658

onnx__MatMul_9659

onnx__MatMul_9684

onnx__MatMul_9685

onnx__MatMul_9686

onnx__MatMul_9687

onnx__MatMul_9688

onnx__MatMul_9689

onnx__MatMul_9690

onnx__MatMul_9715

onnx__MatMul_9716

onnx__MatMul_9717

onnx__MatMul_9718

onnx__MatMul_9722

special_tokens_map.json

tokenizer.json

tokenizer_config.json
```


here's how model-fixed-dims.onnx looks like in netron


```
ONNX v9



onnx.utils.extract_model



0



ai.onnx v14



Extracted from {Extracted from {Extracted from {Extracted from {main_graph}}}}

Inputs



-

name: input_ids

tensor: int64[1,1]



-

name: attention_mask

tensor: int64[1,1024]



-

name: position_ids

tensor: int64[1,1024]



-

name: tree_attention

tensor: float16[1,1,1024,1024]



-

name: past_key_values.0.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.0.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.1.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.1.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.2.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.2.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.3.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.3.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.4.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.4.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.5.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.5.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.6.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.6.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.7.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.7.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.8.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.8.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.9.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.9.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.10.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.10.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.11.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.11.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.12.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.12.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.13.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.13.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.14.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.14.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.15.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.15.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.16.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.16.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.17.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.17.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.18.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.18.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.19.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.19.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.20.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.20.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.21.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.21.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.22.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.22.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.23.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.23.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.24.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.24.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.25.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.25.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.26.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.26.value

tensor: float16[1,2,1023,128]



-

name: past_key_values.27.key

tensor: float16[1,2,1023,128]



-

name: past_key_values.27.value

tensor: float16[1,2,1023,128]

Outputs



-

name: logits

tensor: float16[1,1,151936]



-

name: present.0.key

tensor: float16[1,2,1024,128]



-

name: present.0.value

tensor: float16[1,2,1024,128]



-

name: present.1.key

tensor: float16[1,2,1024,128]



-

name: present.1.value

tensor: float16[1,2,1024,128]



-

name: present.2.key

tensor: float16[1,2,1024,128]



-

name: present.2.value

tensor: float16[1,2,1024,128]



-

name: present.3.key

tensor: float16[1,2,1024,128]



-

name: present.3.value

tensor: float16[1,2,1024,128]



-

name: present.4.key

tensor: float16[1,2,1024,128]



-

name: present.4.value

tensor: float16[1,2,1024,128]



-

name: present.5.key

tensor: float16[1,2,1024,128]



-

name: present.5.value

tensor: float16[1,2,1024,128]



-

name: present.6.key

tensor: float16[1,2,1024,128]



-

name: present.6.value

tensor: float16[1,2,1024,128]



-

name: present.7.key

tensor: float16[1,2,1024,128]



-

name: present.7.value

tensor: float16[1,2,1024,128]



-

name: present.8.key

tensor: float16[1,2,1024,128]



-

name: present.8.value

tensor: float16[1,2,1024,128]



-

name: present.9.key

tensor: float16[1,2,1024,128]



-

name: present.9.value

tensor: float16[1,2,1024,128]



-

name: present.10.key

tensor: float16[1,2,1024,128]



-

name: present.10.value

tensor: float16[1,2,1024,128]



-

name: present.11.key

tensor: float16[1,2,1024,128]



-

name: present.11.value

tensor: float16[1,2,1024,128]



-

name: present.12.key

tensor: float16[1,2,1024,128]



-

name: present.12.value

tensor: float16[1,2,1024,128]



-

name: present.13.key

tensor: float16[1,2,1024,128]



-

name: present.13.value

tensor: float16[1,2,1024,128]



-

name: present.14.key

tensor: float16[1,2,1024,128]



-

name: present.14.value

tensor: float16[1,2,1024,128]



-

name: present.15.key

tensor: float16[1,2,1024,128]



-

name: present.15.value

tensor: float16[1,2,1024,128]



-

name: present.16.key

tensor: float16[1,2,1024,128]



-

name: present.16.value

tensor: float16[1,2,1024,128]



-

name: present.17.key

tensor: float16[1,2,1024,128]



-

name: present.17.value

tensor: float16[1,2,1024,128]



-

name: present.18.key

tensor: float16[1,2,1024,128]



-

name: present.18.value

tensor: float16[1,2,1024,128]



-

name: present.19.key

tensor: float16[1,2,1024,128]



-

name: present.19.value

tensor: float16[1,2,1024,128]



-

name: present.20.key

tensor: float16[1,2,1024,128]



-

name: present.20.value

tensor: float16[1,2,1024,128]



-

name: present.21.key

tensor: float16[1,2,1024,128]



-

name: present.21.value

tensor: float16[1,2,1024,128]



-

name: present.22.key

tensor: float16[1,2,1024,128]



-

name: present.22.value

tensor: float16[1,2,1024,128]



-

name: present.23.key

tensor: float16[1,2,1024,128]



-

name: present.23.value

tensor: float16[1,2,1024,128]



-

name: present.24.key

tensor: float16[1,2,1024,128]



-

name: present.24.value

tensor: float16[1,2,1024,128]



-

name: present.25.key

tensor: float16[1,2,1024,128]



-

name: present.25.value

tensor: float16[1,2,1024,128]



-

name: present.26.key

tensor: float16[1,2,1024,128]



-

name: present.26.value

tensor: float16[1,2,1024,128]



-

name: present.27.key

tensor: float16[1,2,1024,128]



-

name: present.27.value

tensor: float16[1,2,1024,128]







❮

File

Ctrl+O

Open...

Ctrl+Shift+E

Export as PNG

Ctrl+Alt+E

Export as SVG

Edit

Ctrl+F

Find...

View

Ctrl+D

Hide Attributes

Ctrl+I

Hide Weights

Ctrl+U

Hide Names

Ctrl+K

Show Horizontal

Ctrl+M

Mouse Wheel: Zoom

Shift+Up

Zoom In

Shift+Down

Zoom Out

Shift+Backspace

Actual Size

Ctrl+Enter

Properties...

Help

Report Issue

About Netron
```

here's the contents of some files that may help provide additional context

`tokenizer.json` (up until the first few entries of the "vocab" key since the entire vocab is too long to be provided to you)

```json
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 151643,
      "content": "<｜end▁of▁sentence｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151644,
      "content": "<｜User｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151645,
      "content": "<｜Assistant｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151646,
      "content": "<｜begin▁of▁sentence｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151647,
      "content": "<|EOT|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151648,
      "content": "<think>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151649,
      "content": "</think>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151650,
      "content": "<|quad_start|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151651,
      "content": "<|quad_end|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151652,
      "content": "<|vision_start|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151653,
      "content": "<|vision_end|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151654,
      "content": "<|vision_pad|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151655,
      "content": "<|image_pad|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151656,
      "content": "<|video_pad|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151657,
      "content": "<tool_call>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151658,
      "content": "</tool_call>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151659,
      "content": "<|fim_prefix|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151660,
      "content": "<|fim_middle|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151661,
      "content": "<|fim_suffix|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151662,
      "content": "<|fim_pad|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151663,
      "content": "<|repo_name|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151664,
      "content": "<|file_sep|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    }
  ],
  "normalizer": {
    "type": "NFC"
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false
      }
    ]
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {
        "SpecialToken": {
          "id": "<｜begin▁of▁sentence｜>",
          "type_id": 0
        }
      },
      {
        "Sequence": {
          "id": "A",
          "type_id": 0
        }
      }
    ],
    "pair": [
      {
        "SpecialToken": {
          "id": "<｜begin▁of▁sentence｜>",
          "type_id": 0
        }
      },
      {
        "Sequence": {
          "id": "A",
          "type_id": 0
        }
      },
      {
        "SpecialToken": {
          "id": "<｜begin▁of▁sentence｜>",
          "type_id": 1
        }
      },
      {
        "Sequence": {
          "id": "B",
          "type_id": 1
        }
      }
    ],
    "special_tokens": {
      "<｜begin▁of▁sentence｜>": {
        "id": "<｜begin▁of▁sentence｜>",
        "ids": [
          151646
        ],
        "tokens": [
          "<｜begin▁of▁sentence｜>"
        ]
      }
    }
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": false,
    "use_regex": false
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": "",
    "end_of_word_suffix": "",
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {
      "!": 0,
      "\"": 1,
      "#": 2,
      "$": 3,
      "%": 4,
      "&": 5,
      "'": 6,
      "(": 7,
      ")": 8,
      "*": 9,
      "+": 10,
      ",": 11,
      "-": 12,
      ".": 13,
      "/": 14,
      "0": 15,
      "1": 16,
      "2": 17,
```

`tokenizer_config.json`

```json
{
  "add_bos_token": true,
  "add_eos_token": false,
  "add_prefix_space": null,
  "added_tokens_decoder": {
    "151643": {
      "content": "<｜end▁of▁sentence｜>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151644": {
      "content": "<｜User｜>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151645": {
      "content": "<｜Assistant｜>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151646": {
      "content": "<｜begin▁of▁sentence｜>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151647": {
      "content": "<|EOT|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151648": {
      "content": "<think>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151649": {
      "content": "</think>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151650": {
      "content": "<|quad_start|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151651": {
      "content": "<|quad_end|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151652": {
      "content": "<|vision_start|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151653": {
      "content": "<|vision_end|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151654": {
      "content": "<|vision_pad|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151655": {
      "content": "<|image_pad|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151656": {
      "content": "<|video_pad|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151657": {
      "content": "<tool_call>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151658": {
      "content": "</tool_call>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151659": {
      "content": "<|fim_prefix|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151660": {
      "content": "<|fim_middle|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151661": {
      "content": "<|fim_suffix|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151662": {
      "content": "<|fim_pad|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151663": {
      "content": "<|repo_name|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "151664": {
      "content": "<|file_sep|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    }
  },
  "bos_token": "<｜begin▁of▁sentence｜>",
  "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}",
  "clean_up_tokenization_spaces": false,
  "eos_token": "<｜end▁of▁sentence｜>",
  "legacy": true,
  "model_max_length": 16384,
  "pad_token": "<｜end▁of▁sentence｜>",
  "sp_model_kwargs": {},
  "tokenizer_class": "LlamaTokenizer",
  "unk_token": null,
  "use_default_system_prompt": false
}
```

`special_tokens_map.json`

```json
{
  "bos_token": {
    "content": "<｜begin▁of▁sentence｜>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "eos_token": {
    "content": "<｜end▁of▁sentence｜>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "pad_token": {
    "content": "<｜end▁of▁sentence｜>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  }
}
```

huggingface page for the model: https://huggingface.co/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx

here's the text for the page which contains a reference implementation in python

```
Deepseek R1 Distill Qwen 1.5B with Key-Value-Cache enabled in ONNX fp16 format
Model creator: Deepseek
Original model: Deepseek R1 Distill Qwen 1.5B
Description
This repo contains the ONNX files for the ONNX conversion of Deepseek R1 Distill Qwen 1.5B done by Esperanto Technologies. The model is in the fp16 format and has the KVC enabled.

How to download ONNX model and weight files
The easiest way to obtain the model is to clone this whole repo. Alternatively you can download the files is using the huggingface-hub Python library.

pip3 install huggingface-hub>=0.17.1

Then you can download any individual model file to the current directory, at high speed, with a command like this:

huggingface-cli download Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx --local-dir DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx --local-dir-use-symlinks False

For more documentation on downloading with huggingface-cli, please see: HF -> Hub Python Library -> Download files -> Download from the CLI.

How to run from Python code using ONNXRuntime
This model can easily be ran in a CPU using ONNXRuntime.

First install the packages
pip3 install onnx==1.16.1
pip3 install onnxruntime==1.17.1

Example code: generate text with this model
We define the loop with greedy decoding:

import numpy as np
import onnxruntime
import onnx
from transformers import AutoTokenizer

def generate_text(model_path, prompt, tokenizer, max_gen_tokens, total_sequence, window, context):
    model = onnx.load(model_path)

    #we create the inputs for the first iteration
    input_tensor = tokenizer(prompt, return_tensors="pt")
    prompt_size = len(input_tensor['input_ids'][0])
    actual_input = input_tensor['input_ids']
    if prompt_size < window:
        actual_input = np.concatenate((tokenizer.bos_token_id*np.ones([1, window - prompt_size], dtype = 'int64'),
                                       actual_input), axis=1)
    if prompt_size + max_gen_tokens > total_sequence:
        print("ERROR: Longer total sequence is needed!")
        return
    first_attention = np.concatenate((np.zeros([1, total_sequence - window], dtype = 'int64'),
                                      np.ones((1, window), dtype = 'int64')), axis=1)
    max_gen_tokens += prompt_size #we need to generate on top of parsing the prompt
    inputs_names =[node.name for node in model.graph.input]
    output_names =[node.name for node in model.graph.output]
    inputs_dict = {}
    inputs_dict['input_ids'] = actual_input[:, :window].reshape(1, window).numpy()
    inputs_dict['attention_mask'] = first_attention
    index_pos = sum(first_attention[0])
    inputs_dict['position_ids'] = np.concatenate((np.zeros([1, total_sequence - index_pos], dtype = 'int64'), np.arange(index_pos, dtype = 'int64').reshape(1, index_pos)), axis=1)
    inputs_dict['tree_attention'] = np.triu(-65504*np.ones(total_sequence), k= 1).astype('float16').reshape(1, 1, total_sequence, total_sequence)
    for name in inputs_names:
        if name == 'input_ids' or name == 'attention_mask' or name == 'position_ids' or name == 'tree_attention': continue
        inputs_dict[name] = np.zeros([1, 2, context-window, 64], dtype="float16")
    index = 0
    new_token = np.array([10])
    next_index = window
    old_j = 0
    total_input = actual_input.numpy()

    rt_session = onnxruntime.InferenceSession(model_path)
    ## We run the inferences
    while next_index < max_gen_tokens:
        if new_token.any() == tokenizer.eos_token_id:
            break
        #inference
        output = rt_session.run(output_names, inputs_dict)
        outs_dictionary = {name: content for (name, content) in zip (output_names, output)}
        #we prepare the inputs for the next inference
        for name in inputs_names:
            if name == 'input_ids':
                old_j = next_index
                if next_index < prompt_size:
                    if prompt_size - next_index >= window: next_index += window
                    else: next_index = prompt_size 
                    j = next_index - window
                else:
                    next_index +=1
                    j = next_index - window
                    new_token = outs_dictionary['logits'].argmax(-1).reshape(1, window)
                    total_input = np.concatenate((total_input, new_token[: , -1:]), axis = 1)
                inputs_dict['input_ids']= total_input[:, j:next_index].reshape(1, window)
            elif name == 'attention_mask':
                inputs_dict['attention_mask'] = np.concatenate((np.zeros((1, total_sequence-next_index), dtype = 'int64'), np.ones((1, next_index), dtype = 'int64')), axis=1)
            elif name == 'position_ids':
                inputs_dict['position_ids'] = np.concatenate((np.zeros([1, total_sequence - next_index], dtype = 'int64'), np.arange(next_index, dtype = 'int64').reshape(1, next_index)), axis=1)
            elif name == 'tree_attention': continue
            else:
                old_name = name.replace("past_key_values", "present")
                inputs_dict[name] = outs_dictionary[old_name][:, next_index-old_j:context-window+(next_index - old_j), :]

    answer = tokenizer.decode(total_input[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answer

We now run the inferences:

tokenizer = AutoTokenizer.from_pretrained("Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx")
model_path = "DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/model.onnx"

max_gen_tokens = 20    #number of tokens we want tog eneral
total_sequence = 128   #total sequence_length
context = 1024         #the context to extend the kvc
window = 16            #number of tokens we want to parse at the time
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

generated = generate_text(model_path, prompt, tokenizer, max_gen_tokens, total_sequence, window, context)
print(generated)
```

can you identify the issues and give me a complete, working src/main.rs implementation that performs inference correctly?