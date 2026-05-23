use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
struct Args {
    input: PathBuf,
    out_dir: PathBuf,
    dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct BenchDump {
    model_path: String,
    model_id: String,
    n_warmup: usize,
    max_new_tokens: usize,
    templates: BTreeMap<String, TemplateInputs>,
}

#[derive(Debug, Deserialize)]
struct TemplateInputs {
    forced: Vec<Vec<u32>>,
    plain: Vec<Vec<u32>>,
    prefix: Vec<u32>,
    verdict_token_ids: Vec<u32>,
    expected_verdicts: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptLevel {
    L0,
    L1,
    L3,
}

impl OptLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::L0 => "L0",
            Self::L1 => "L1",
            Self::L3 => "L3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuffixPath {
    Chunked,
    TokenByToken,
}

impl SuffixPath {
    fn as_str(self) -> &'static str {
        match self {
            Self::Chunked => "chunked",
            Self::TokenByToken => "token-by-token",
        }
    }
}

#[derive(Debug, Serialize)]
struct BenchResult {
    variant: &'static str,
    runtime: &'static str,
    model_id: String,
    device: &'static str,
    dtype: &'static str,
    provider: &'static str,
    n_samples: usize,
    n_warmup: usize,
    input_token_count_median: usize,
    output_token_count: usize,
    latency: LatencyStats,
    extra: Extra,
    timestamp_utc: String,
    host: String,
    torch_version: Option<String>,
}

#[derive(Debug, Serialize)]
struct Extra {
    mode: &'static str,
    opt_level: String,
    runtime: &'static str,
    precision: &'static str,
    template: String,
    kv_cache: bool,
    threads: Option<usize>,
    suffix_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LatencyStats {
    n: usize,
    mean_ms: f64,
    stdev_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
    throughput_rps: f64,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let dump = read_dump(&args.input)?;
    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("failed to create {}", args.out_dir.display()))?;

    // --dry-run is a real but tiny run — it still loads the model and runs the
    // candle forward, so the whole path is smoke-tested; only the counts shrink.
    let (n_warmup, sample_cap, max_new_tokens) = if args.dry_run {
        println!("dry-run: 2 samples, 1 warmup, max_new_tokens=4");
        (1usize, 2usize, 4usize)
    } else {
        (dump.n_warmup, usize::MAX, dump.max_new_tokens)
    };

    let device = Device::Cpu;
    let mut model = load_model(&dump.model_path, &device)?;

    for (template_name, template) in &dump.templates {
        // L2 +kv: prime the shared system-prompt prefix once, clone the primed
        // state per sample, feed only the suffix. Same prefix-KV trick the
        // ONNX +kv and llama.cpp +kv cells use.
        let l2_kv = run_l2(
            &mut model,
            &device,
            &dump,
            template_name,
            template,
            n_warmup,
            template.forced.len().min(sample_cap),
        )?;
        write_result(&args.out_dir, l2_kv)?;

        // L2 (no-kv): feed the full forced sequence per sample, no priming.
        // The matched baseline for the +kv comparison.
        let l2_nokv = run_l2_no_kv(
            &mut model,
            &device,
            &dump,
            template_name,
            template,
            n_warmup,
            template.forced.len().min(sample_cap),
        )?;
        write_result(&args.out_dir, l2_nokv)?;

        let l0 = run_l0(
            &mut model,
            &device,
            &dump,
            template_name,
            template,
            n_warmup,
            template.plain.len().min(sample_cap),
            max_new_tokens,
        )?;
        write_result(&args.out_dir, l0)?;
    }

    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut input = None;
    let mut out_dir = None;
    let mut dry_run = false;
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| anyhow!("--input requires a path"))?,
                ));
            }
            "--out-dir" => {
                out_dir = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| anyhow!("--out-dir requires a path"))?,
                ));
            }
            "--dry-run" => dry_run = true,
            "-h" | "--help" => {
                println!("qwen3guard-bench --input <json> --out-dir <dir> [--dry-run]");
                std::process::exit(0);
            }
            _ => bail!("unknown argument: {arg}"),
        }
    }

    Ok(Args {
        input: input.ok_or_else(|| anyhow!("missing --input <json>"))?,
        out_dir: out_dir.ok_or_else(|| anyhow!("missing --out-dir <dir>"))?,
        dry_run,
    })
}

fn read_dump(path: &Path) -> Result<BenchDump> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read input dump {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("failed to parse {}", path.display()))
}

fn load_model(model_path: &str, device: &Device) -> Result<ModelForCausalLM> {
    let config_path = Path::new(model_path).join("config.json");
    let config_data = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let cfg: Config = serde_json::from_str(&config_data)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    let weights = Path::new(model_path).join("model.safetensors");
    let weights = weights
        .to_str()
        .ok_or_else(|| anyhow!("model path is not valid UTF-8"))?
        .to_string();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
    Ok(ModelForCausalLM::new(&cfg, vb)?)
}

/// Run `n_warmup` warmup iterations then `n_samples` timed iterations of
/// `run_one`, returning per-iteration latencies in milliseconds.
fn measure<F>(n_warmup: usize, n_samples: usize, mut run_one: F) -> Result<Vec<f64>>
where
    F: FnMut(usize) -> Result<()>,
{
    for i in 0..n_warmup {
        run_one(i % n_samples)?;
    }
    let mut latencies = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let start = Instant::now();
        run_one(i)?;
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(latencies)
}

fn run_l2(
    model: &mut ModelForCausalLM,
    device: &Device,
    dump: &BenchDump,
    template_name: &str,
    template: &TemplateInputs,
    n_warmup: usize,
    n_samples: usize,
) -> Result<BenchResult> {
    ensure_template_l2(template_name, template)?;
    model.clear_kv_cache();
    let prefix = input_tensor(&template.prefix, device)?;
    model.forward(&prefix, 0)?;
    let primed = model.clone();

    let (suffix_path, verify_count) = choose_l2_path(&primed, device, template_name, template)?;
    println!(
        "{template_name} L2 suffix_path={} verify {verify_count}/10",
        suffix_path.as_str()
    );

    let prefix_len = template.prefix.len();
    let latencies = measure(n_warmup, n_samples, |i| {
        let suffix = &template.forced[i][prefix_len..];
        let mut sample_model = primed.clone();
        predict_l2(
            &mut sample_model,
            device,
            suffix,
            prefix_len,
            &template.verdict_token_ids,
            suffix_path,
        )?;
        Ok(())
    })?;

    let counts: Vec<usize> = template
        .forced
        .iter()
        .take(n_samples)
        .map(Vec::len)
        .collect();
    Ok(make_result(
        dump,
        template_name,
        OptLevel::L3,
        true,
        n_samples,
        n_warmup,
        median_usize(&counts),
        1,
        LatencyStats::from_ms(&latencies),
        Some(suffix_path),
    ))
}

fn choose_l2_path(
    primed: &ModelForCausalLM,
    device: &Device,
    template_name: &str,
    template: &TemplateInputs,
) -> Result<(SuffixPath, usize)> {
    match verify_l2_path(primed, device, template, SuffixPath::Chunked) {
        Ok(k) if k >= 9 => return Ok((SuffixPath::Chunked, k)),
        Ok(k) => eprintln!(
            "{template_name} L2 chunked verify {k}/10; falling back to token-by-token"
        ),
        Err(err) => eprintln!(
            "{template_name} L2 chunked failed: {err}; falling back to token-by-token"
        ),
    }
    let fallback_k = verify_l2_path(primed, device, template, SuffixPath::TokenByToken)?;
    Ok((SuffixPath::TokenByToken, fallback_k))
}

fn verify_l2_path(
    primed: &ModelForCausalLM,
    device: &Device,
    template: &TemplateInputs,
    suffix_path: SuffixPath,
) -> Result<usize> {
    let n = template
        .expected_verdicts
        .len()
        .min(10)
        .min(template.forced.len());
    let mut correct = 0usize;

    for i in 0..n {
        let sample = &template.forced[i];
        let suffix = &sample[template.prefix.len()..];
        let mut sample_model = primed.clone();
        let pred = predict_l2(
            &mut sample_model,
            device,
            suffix,
            template.prefix.len(),
            &template.verdict_token_ids,
            suffix_path,
        )?;
        if pred == template.expected_verdicts[i] {
            correct += 1;
        }
    }

    Ok(correct)
}

fn run_l2_no_kv(
    model: &mut ModelForCausalLM,
    device: &Device,
    dump: &BenchDump,
    template_name: &str,
    template: &TemplateInputs,
    n_warmup: usize,
    n_samples: usize,
) -> Result<BenchResult> {
    ensure_template_l2(template_name, template)?;
    // Verify the no-kv path against the PyTorch-L2 oracle on the first 10
    // samples before the timed loop. Catches a wrong-position-offset or
    // wrong-tokens-fed bug loudly instead of letting silent drift through.
    let verify_n = 10.min(template.forced.len()).min(template.expected_verdicts.len());
    let mut agree = 0usize;
    for i in 0..verify_n {
        model.clear_kv_cache();
        let pred = predict_l2(
            model,
            device,
            &template.forced[i],
            0,
            &template.verdict_token_ids,
            SuffixPath::Chunked,
        )?;
        if pred == template.expected_verdicts[i] {
            agree += 1;
        }
    }
    println!("{template_name} L2 no-kv verify {agree}/{verify_n}");
    if agree < verify_n {
        bail!("{template_name} L2 no-kv: verdict mismatch vs PyTorch-L2 oracle ({agree}/{verify_n})");
    }

    // No priming: every sample starts from an empty KV cache and feeds the
    // full forced_ids (prefix + user content + "Safety: "). The chunked path
    // is used (matches the +kv path's kernel sequence after the prefix).
    let latencies = measure(n_warmup, n_samples, |i| {
        model.clear_kv_cache();
        predict_l2(
            model,
            device,
            &template.forced[i],
            0,
            &template.verdict_token_ids,
            SuffixPath::Chunked,
        )?;
        Ok(())
    })?;

    let counts: Vec<usize> = template
        .forced
        .iter()
        .take(n_samples)
        .map(Vec::len)
        .collect();
    Ok(make_result(
        dump,
        template_name,
        OptLevel::L1,
        false,
        n_samples,
        n_warmup,
        median_usize(&counts),
        1,
        LatencyStats::from_ms(&latencies),
        Some(SuffixPath::Chunked),
    ))
}

fn run_l0(
    model: &mut ModelForCausalLM,
    device: &Device,
    dump: &BenchDump,
    template_name: &str,
    template: &TemplateInputs,
    n_warmup: usize,
    n_samples: usize,
    max_new_tokens: usize,
) -> Result<BenchResult> {
    ensure_template_l0(template_name, template)?;
    let latencies = measure(n_warmup, n_samples, |i| {
        run_l0_sample(model, device, &template.plain[i], max_new_tokens)
    })?;

    let counts: Vec<usize> = template
        .plain
        .iter()
        .take(n_samples)
        .map(Vec::len)
        .collect();
    Ok(make_result(
        dump,
        template_name,
        OptLevel::L0,
        false,
        n_samples,
        n_warmup,
        median_usize(&counts),
        max_new_tokens,
        LatencyStats::from_ms(&latencies),
        None,
    ))
}

fn run_l0_sample(
    model: &mut ModelForCausalLM,
    device: &Device,
    prompt: &[u32],
    max_new_tokens: usize,
) -> Result<()> {
    model.clear_kv_cache();
    let input = input_tensor(prompt, device)?;
    let logits = model.forward(&input, 0)?;
    let mut next = argmax_full(&last_logits(&logits)?)? as u32;
    let mut offset = prompt.len();

    // The prefill forward above already produced the first token; decode the
    // remaining max_new_tokens-1 so the loop yields max_new_tokens in total.
    for _ in 1..max_new_tokens {
        let input = input_tensor(&[next], device)?;
        let logits = model.forward(&input, offset)?;
        next = argmax_full(&last_logits(&logits)?)? as u32;
        offset += 1;
    }

    Ok(())
}

fn predict_l2(
    model: &mut ModelForCausalLM,
    device: &Device,
    suffix: &[u32],
    offset: usize,
    verdict_token_ids: &[u32],
    suffix_path: SuffixPath,
) -> Result<usize> {
    if suffix.is_empty() {
        bail!("empty L2 suffix after prefix")
    }

    let logits = match suffix_path {
        SuffixPath::Chunked => {
            let input = input_tensor(suffix, device)?;
            model.forward(&input, offset)?
        }
        SuffixPath::TokenByToken => {
            let mut last = None;
            for (i, token) in suffix.iter().enumerate() {
                let input = input_tensor(&[*token], device)?;
                last = Some(model.forward(&input, offset + i)?);
            }
            last.ok_or_else(|| anyhow!("empty token-by-token suffix"))?
        }
    };

    argmax_verdict(&last_logits(&logits)?, verdict_token_ids)
}

fn input_tensor(tokens: &[u32], device: &Device) -> Result<Tensor> {
    if tokens.is_empty() {
        bail!("cannot build an empty input tensor")
    }
    Ok(Tensor::from_vec(
        tokens.to_vec(),
        (1, tokens.len()),
        device,
    )?)
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    match dims.len() {
        1 => Ok(logits.clone()),
        2 => logits.i((dims[0] - 1, ..)).map_err(Into::into),
        3 => {
            if dims[0] != 1 {
                bail!("expected batch size 1, got {}", dims[0])
            }
            logits.i((0, dims[1] - 1, ..)).map_err(Into::into)
        }
        _ => bail!("unsupported logits shape: {dims:?}"),
    }
}

fn argmax_verdict(logits: &Tensor, verdict_token_ids: &[u32]) -> Result<usize> {
    if verdict_token_ids.len() != 3 {
        bail!("expected exactly 3 verdict token ids")
    }
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;

    for (idx, &token_id) in verdict_token_ids.iter().enumerate() {
        let value = logits
            .i(token_id as usize)
            .with_context(|| format!("verdict token id {token_id} outside logits vocab"))?
            .to_scalar::<f32>()?;
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }

    Ok(best_idx)
}

fn argmax_full(logits: &Tensor) -> Result<usize> {
    Ok(logits.argmax(0usize)?.to_scalar::<u32>()? as usize)
}

fn make_result(
    dump: &BenchDump,
    template: &str,
    opt_level: OptLevel,
    kv_cache: bool,
    n_samples: usize,
    n_warmup: usize,
    input_token_count_median: usize,
    output_token_count: usize,
    latency: LatencyStats,
    suffix_path: Option<SuffixPath>,
) -> BenchResult {
    BenchResult {
        variant: "gen",
        runtime: "rust-candle",
        model_id: dump.model_id.clone(),
        device: "cpu",
        dtype: "fp32",
        provider: "candle-cpu",
        n_samples,
        n_warmup,
        input_token_count_median,
        output_token_count,
        latency,
        extra: Extra {
            mode: "representative",
            opt_level: opt_level.as_str().to_string(),
            runtime: "rust-candle",
            precision: "fp32",
            template: template.to_string(),
            kv_cache,
            threads: None,
            suffix_path: suffix_path.map(|path| path.as_str().to_string()),
        },
        timestamp_utc: timestamp(),
        host: host_name(),
        torch_version: None,
    }
}

fn write_result(out_dir: &Path, result: BenchResult) -> Result<()> {
    let kv_tag = if result.extra.kv_cache { "_kvcache" } else { "" };
    let filename = format!(
        "bench_gen_rust-candle_cpu_{}_{}{}_{}.json",
        result.extra.template,
        result.extra.opt_level,
        kv_tag,
        timestamp()
    );
    let path = out_dir.join(filename);
    let data = serde_json::to_string_pretty(&result)?;
    fs::write(&path, data).with_context(|| format!("failed to write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

impl LatencyStats {
    fn from_ms(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                n: 0,
                mean_ms: 0.0,
                stdev_ms: 0.0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                throughput_rps: 0.0,
            };
        }

        let n = values.len();
        let mean_ms = values.iter().sum::<f64>() / n as f64;
        let stdev_ms = (values
            .iter()
            .map(|value| {
                let delta = value - mean_ms;
                delta * delta
            })
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let min_ms = values
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, value| acc.min(value));
        let max_ms = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));

        Self {
            n,
            mean_ms,
            stdev_ms,
            p50_ms: percentile_nearest_rank(values, 50.0),
            p95_ms: percentile_nearest_rank(values, 95.0),
            p99_ms: percentile_nearest_rank(values, 99.0),
            min_ms,
            max_ms,
            throughput_rps: if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 },
        }
    }
}

fn percentile_nearest_rank(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = ((percentile / 100.0) * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn median_usize(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn ensure_template_l2(template_name: &str, template: &TemplateInputs) -> Result<()> {
    if template.forced.is_empty() {
        bail!("{template_name}: forced inputs are empty")
    }
    if template.prefix.is_empty() {
        bail!("{template_name}: prefix is empty")
    }
    for (idx, sample) in template.forced.iter().enumerate() {
        if sample.len() <= template.prefix.len() {
            bail!("{template_name}: forced[{idx}] does not have a suffix")
        }
        if !sample.starts_with(&template.prefix) {
            bail!("{template_name}: forced[{idx}] does not start with prefix")
        }
    }
    Ok(())
}

fn ensure_template_l0(template_name: &str, template: &TemplateInputs) -> Result<()> {
    if template.plain.is_empty() {
        bail!("{template_name}: plain inputs are empty")
    }
    Ok(())
}

fn host_name() -> String {
    env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string())
}

/// Unix epoch seconds — orders result files (latest wins) and keeps result
/// filenames unique across runs.
fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}
