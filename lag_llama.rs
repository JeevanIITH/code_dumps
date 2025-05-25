//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! Implementation based on Hugging Face's [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

use super::{deepseek2::SplitOp, with_tracing::{linear_no_bias as linear, Linear, RmsNorm}};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Init, Module, VarBuilder};
use num_traits::Pow;
use std::{collections::HashMap, f32::consts::PI};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;


#[derive(Debug, Clone)]
pub struct Config {
    pub feature_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd_per_head: usize,
    pub dropout: f32,
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    position_ids: &Tensor, // shape: [batch_size, seq_len]
) -> Result<(Tensor, Tensor)> {
    // Squeeze cos/sin from shape [1, 1, seq_len, dim] to [seq_len, dim]
    let cos = cos.squeeze(1)?.squeeze(0)?; // [seq_len, dim]
    let sin = sin.squeeze(1)?.squeeze(0)?; // [seq_len, dim]

    // Index cos/sin with position_ids -> [batch_size, seq_len, dim]
    let cos = cos.index_select(position_ids, 0)?; // [bs, seq_len, dim]
    let sin = sin.index_select(position_ids, 0)?; // [bs, seq_len, dim]

    // Unsqueeze to [bs, 1, seq_len, dim]
    let cos = cos.unsqueeze(1)?; 
    let sin = sin.unsqueeze(1)?;

    // Apply rotary embeddings
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin))?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin))?;
    Ok((q_embed, k_embed))
}

#[derive(Debug, Clone)]
pub struct LlamaRotaryEmbedding {
    dim: usize,
    max_position_embeddings: usize,
    base: f32,
    inv_freq: Tensor,
    cos_cached: Tensor,
    sin_cached: Tensor,
}

impl LlamaRotaryEmbedding {
    pub fn new(dtype: DType,dim: usize,  max_position_embeddings: usize, base: f32, device: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (base.powf(i as f32/dim as f32)))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1,inv_freq_len), device)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_position_embeddings,1))?;

        let freqs = t.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs,&freqs], D::Minus1)?;
        let cos_cached = freqs.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        let sin_cached = freqs.sin()?.unsqueeze(0)?.unsqueeze(0)?;
        // Self { dim: , max_position_embeddings: (), base: (), inv_freq: (), cos_cached: (), sin_cache: () }
        Ok(Self { 
            dim, 
            max_position_embeddings, 
            base, 
            inv_freq, 
            cos_cached, 
            sin_cached,
        })
    }

    pub fn forward(&mut self, seq_len: usize, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        if seq_len > self.max_position_embeddings {
            let t = Tensor::arange(0u32, seq_len as u32, device)?
                .to_dtype(dtype)?
                .reshape((seq_len,1))?;
            let freqs = t.matmul(&self.inv_freq)?;
            let freqs = Tensor::cat(&[&freqs,&freqs], D::Minus1)?;
            let cos_cached = freqs.cos()?.unsqueeze(0)?.unsqueeze(0)?;
            let sin_cached = freqs.sin()?.unsqueeze(0)?.unsqueeze(0)?;
            self.cos_cached = cos_cached;
            self.sin_cached = sin_cached;
            
        }
        let cos_cached = self.cos_cached.narrow(0, 0, seq_len)?;
        let sin_cached = self.sin_cached.narrow(0, 0, seq_len)?;
        Ok((cos_cached,sin_cached))
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    kv_proj: Linear,    
    o_proj: Linear,
    n_head: usize,
    n_embd_per_head: usize,
    block_size: usize,
    dropout: f32,
    pub kv_cache: Option<(Tensor,Tensor)>,
    rotary_emb: Option<LlamaRotaryEmbedding>,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {

    fn forward_2(
        &mut self,
        x: &Tensor,             // [B, T, C]
        use_kv_cache: bool,
    ) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?; // B, T, C
    
        // Project queries, keys, and values
        let q = self.q_proj.forward(x)?;
        let kv = self.kv_proj.forward(x)?;
        // let kv_mix = kv.split(&[c], 2)?; // Split on last dim
        // let (k,v) = (kv_mix[0],kv_mix[1]);
        let k = kv.narrow(2, 0, c)?;         // take first half [B, T, C]
        let v = kv.narrow(2, c, c)?;
    
        // let cache_initialized = cache.kvs[block_idx].is_some();
        let cache_initialized = self.kv_cache.is_some();

        let ( k, v) = if use_kv_cache {
            if cache_initialized {
                let (prev_k, prev_v) = self.kv_cache.as_ref().unwrap();
                let k = Tensor::cat(&[prev_k, &k], 1)?.narrow(1, 1, prev_k.dim(1)?)?;
                let v = Tensor::cat(&[prev_v, &v], 1)?.narrow(1, 1, prev_v.dim(1)?)?;
                // cache.kvs[block_idx] = Some((k.clone(), v.clone()));
                self.kv_cache = Some((k.clone(),v.clone()));
                (k, v)
            } else {
                // cache.kvs[block_idx] = Some((k.clone(), v.clone()));
                self.kv_cache = Some((k.clone(),v.clone()));
                (k, v)
            }
        } else {
            (k, v)
        };
    
        // Reshape to [B, nh, T, hs]
        let mut q = q
            .reshape((b, t, self.n_head, self.n_embd_per_head))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((b, t, self.n_head, self.n_embd_per_head))?
            .transpose(1, 2)?;
        // let v = v
        //     .reshape((b, v.dim(1)? / (self.num_attention_heads * self.head_dim), self.num_attention_heads, self.head_dim))?
        //     .transpose(1, 2)?;
        let mut v = v
            .reshape((b, t, self.n_head, self.n_embd_per_head))?
            .transpose(1, 2)?;
    
        let true_seq_len = k.dim(2)?; // after cache concat
    
        // Rotary Embeddings
        if let Some(rotary_emb) = self.rotary_emb.as_mut() {
            if use_kv_cache && cache_initialized {
                let (cos,sin) = rotary_emb.forward(true_seq_len, v.device(), v.dtype()).unwrap();
                // let position_ids = Tensor::new(&[-1], &x.device())?.to_dtype(DType::U32)?;
                let (_, _, t_last, _) = q.dims4()?; // q: [B, H, T, D]
                let position_id = Tensor::arange((t_last - 1) as u32, t_last as u32, &x.device())?.to_dtype(DType::U32)?;

                let (q_rot, _) = apply_rotary_pos_emb(&q, &k, &cos, &sin, &position_id)?;
                let (_, k_rot) = apply_rotary_pos_emb(&q, &k, &cos, &sin, &Tensor::new(&[] as &[u32], &x.device())?)?;
                (q, k) = (q_rot, k_rot);
            }
            else {
                let (cos,sin) = rotary_emb.forward(t, v.device(), v.dtype()).unwrap();
                let position_ids = Tensor::arange(0, t as u32, &x.device())?.to_dtype(DType::U32)?;
                let (q_rot, k_rot) = apply_rotary_pos_emb(&q, &k, &cos, &sin, &position_ids)?;
                (q, k) = (q_rot, k_rot);
            }
        }
        // if let Ok((cos, sin)) = self.rotary_emb.forward(true_seq_len, &v.device(), v.dtype()).unwrap() {
        //     if use_kv_cache && cache_initialized {
        //         let position_ids = Tensor::new(&[-1], &x.device())?.to_dtype(DType::U32)?;
        //         let (q_rot, _) = apply_rotary_pos_emb(&q, &k, &cos, &sin, &position_ids)?;
        //         let (_, k_rot) = apply_rotary_pos_emb(&q, &k, &cos, &sin, &Tensor::new(&[], &x.device())?)?;
        //         (q, k) = (q_rot, k_rot);
        //     } else {
        //         let position_ids = Tensor::arange(0, t as u32, &x.device())?.to_dtype(DType::U32)?;
        //         let (q_rot, k_rot) = apply_rotary_pos_emb(&q, &k, &cos, &sin, &position_ids)?;
        //         (q, k) = (q_rot, k_rot);
        //     }
        // }
    
        // Scaled Dot-Product Attention (Flash or default)
        let y = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let scale = 1.0 / (self.n_embd_per_head as f32).sqrt();
            flash_attn(&q, &k, &v, scale, !use_kv_cache || !cache_initialized)?.transpose(1, 2)?
        } else {
            let qf = q.to_dtype(DType::F32)?;
            let kf = k.to_dtype(DType::F32)?;
            let vf = v.to_dtype(DType::F32)?;
            let attn = (qf.matmul(&kf.t()?)? / (self.n_embd_per_head as f64).sqrt())?;
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            attn.matmul(&vf.contiguous()?)?.to_dtype(q.dtype())?
        };
    
        let y = y.transpose(1, 2)?.reshape(&[b, t, c])?;
        self.o_proj.forward(&y)
    }
    // fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
    //     crate::utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
    // }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        // let size_in = cfg.hidden_size;
        // let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        // let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(
                cfg.n_embd_per_head * cfg.n_head,
                cfg.n_embd_per_head * cfg.n_head, 
                vb.pp("q_proj")
        )?;
        let kv_proj = linear(
                cfg.n_embd_per_head * cfg.n_head,
                2 * cfg.n_embd_per_head * cfg.n_head, 
                vb.pp("kv_proj")
        )?;
        
        // let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(
            cfg.n_embd_per_head * cfg.n_head, 
            cfg.n_embd_per_head * cfg.n_head, 
            vb.pp("o_proj")
        )?;
        
        let n_head = cfg.n_head;
        let n_embd_per_head = cfg.n_embd_per_head;
        let block_size = cfg.block_size;
        let dropout = cfg.dropout;
        let rotary_emb = Some(LlamaRotaryEmbedding::new(vb.dtype, n_embd_per_head, block_size, 10000.0, vb.device())?);
        let use_flash_attn = false;
        Ok(Self { q_proj, 
            kv_proj, 
            o_proj, 
            n_head, 
            n_embd_per_head, 
            block_size, 
            dropout, 
            kv_cache: None, 
            rotary_emb, 
            use_flash_attn, 
            span, 
            span_rot, 
        })        
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}


fn find_multiple(n: usize, k: usize) -> usize{
    if n % k == 0 {
        return n;
    }
    return n + k - ( n % k );
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        // let h_size = cfg.hidden_size;
        // let i_size = cfg.intermediate_size;
        let hidden_dim = 4 * cfg.n_embd_per_head * cfg.n_head;
        let n_hidden = find_multiple(2*hidden_dim/3, 256);
        let c_fc1 = linear(cfg.n_embd_per_head * cfg.n_head, n_hidden, vb.pp("gate_proj"))?;
        // let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_fc2 = linear(cfg.n_embd_per_head * cfg.n_head, n_hidden, vb.pp("gate_proj"))?;
        let c_proj = linear(n_hidden, cfg.n_embd_per_head * cfg.n_head, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

pub struct RMSNorm {
    scale: Tensor, // equivalent to nn.Parameter in PyTorch
    eps: f64,
    dim: usize,
}

impl RMSNorm {
    pub fn new(size: usize, dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get_with_hints((size,), "scale", Init::Const(1.0))?;
        Ok(Self { scale, eps, dim })
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Ensure computation in float32
        let x_fp32 = x.to_dtype(DType::F32)?;
        let mean_sq = x_fp32.sqr()?.mean_keepdim(self.dim)?;
        // let rms_inv = (mean_sq + self.eps)?.rsqrt()?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let rms_inv = rms.recip()?;
        let x_normed = x_fp32.mul(&rms_inv)?;
        let scaled = x_normed.broadcast_mul(&self.scale)?;
        scaled.to_dtype(x.dtype())
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &mut self,
        x: &Tensor,
        use_kv_cache: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        // let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward_2(&x, use_kv_cache))?;

        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    context_length: usize,
    lags_seq: Vec<usize>,
    num_parallel_samples: usize,
    scaler: Box<dyn Scaler>
    distr_output: DistributionOutput,
    param_proj: Box<dyn Module>,

    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,

    y_cache: bool,
}

impl Llama {
    // required by LLaVA
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }
    // required by LLaVA
    pub fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn forward(
        &self, 
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Option<Tensor>,
        future_time_feat: Option<Tensor>,
        future_target: Option<Tensor>,
        use_kv_cache: bool,
        x: &Tensor, 
        index_pos: usize, 
    ) -> Result<Tensor> {
        

        let (mut transformer_input, loc, scale) = self.prepare_input(
            &past_target, 
            &past_observed_values, 
            past_time_feat, 
            future_time_feat, 
            future_target
        );

        if use_kv_cache && self.y_cache {
            // Slice the last time step along the sequence dimension (assume dim=1 for shape [batch, seq_len, ...])
            transformer_input = transformer_input.narrow(1, transformer_input.dim(1)? - 1, 1)?;
        }

        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        if use_kv_cache {
            self.y_cache = true;
        }
        let params = self.param_proj(x)?;
        return (params , loc , scale);
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(
        vb: VarBuilder, 
        cfg: &Config, 
        context_length: usize,
        max_context_length: usize,
        scaling: &str,
        input_size: usize,
        lags_seq: Vec<usize>,
        distr_output: DistributionOutput,
        num_parallel_samples: usize,
        time_feat: bool,
        dropout: f32,
    ) -> Result<Self> {

        let feature_size = if time_feat {
            input_size * lags_seq.len() + 2 * input_size + 6
        } else {
            input_size * lags_seq.len() + 2 * input_size
        };

        let scaler: Box<dyn Scaler> = match scaling {
            // TODO impl scaler here.
        };

        let param_proj = distr_output.get_args_proj(cfg.n_embd_per_head * cfg.n_head)?;

        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }

    fn prepare_input(
        &self,
        past_target: &Tensor,
        past_observed_values: &Tensor,
        past_time_feat: Option<&Tensor>,
        future_time_feat: Option<&Tensor>,
        future_target: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (scaled_past_target, loc, scale) = self.scaler(past_target, past_observed_values)?;
    
        let max_lag = *self.lags_seq.iter().max().unwrap();
    
        let input = if let Some(future_target) = future_target {
            let context = scaled_past_target.narrow(D::Minus1, max_lag, scaled_past_target.dim(D::Minus1)? - max_lag)?;
            let future_scaled = (future_target.narrow(D::Minus1, 0, future_target.dim(D::Minus1)? - 1)? - &loc)?
                .broadcast_div(&scale)?;
            Tensor::cat(&[&context, &future_scaled], D::Minus1)?
        } else {
            scaled_past_target.narrow(D::Minus1, max_lag, scaled_past_target.dim(D::Minus1)? - max_lag)?
        };
    
        let time_feat = if let (Some(past_time), Some(future_time)) = (past_time_feat, future_time_feat) {
            let past = past_time.narrow(D::Minus2, max_lag, past_time.dim(D::Minus2)? - max_lag)?;
            let future = future_time.narrow(D::Minus2, 0, future_time.dim(D::Minus2)? - 1)?;
            Some(Tensor::cat(&[&past, &future], D::Minus2)?)
        } else {
            past_time_feat.map(|pt| pt.narrow(D::Minus2, max_lag, pt.dim(D::Minus2)? - max_lag))
        };
    
        let prior_input = (past_target.narrow(D::Minus1, 0, max_lag)? - &loc)?.broadcast_div(&scale)?;
    
        let lags = lagged_sequence_values(&self.lags_seq, &prior_input, &input, D::Minus1)?;
    
        let static_feat = Tensor::cat(
            &[
                &loc.abs()?.log1p()?,
                &scale.log()?,
            ],
            D::Minus1,
        )?;
    
        let expanded_static_feat = unsqueeze_expand(&static_feat, D::Minus2, lags.dim(D::Minus2)?)?;
    
        let full_feat = if let Some(time_feat) = time_feat {
            Tensor::cat(&[&lags, &expanded_static_feat, &time_feat], D::Minus1)?
        } else {
            Tensor::cat(&[&lags, &expanded_static_feat], D::Minus1)?
        };
    
        Ok((full_feat, loc, scale))
    }
}
