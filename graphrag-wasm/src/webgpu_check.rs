//! WebGPU support detection and capability checking
//!
//! This module provides utilities to detect WebGPU availability
//! and query GPU capabilities for optimal ML inference.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, window};

/// WebGPU capability information
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebGPUInfo {
    /// Whether WebGPU is available
    pub available: bool,
    /// GPU vendor name (e.g., "nvidia", "amd", "intel")
    vendor: String,
    /// GPU architecture (e.g., "ampere", "rdna2")
    architecture: String,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Maximum texture dimension
    pub max_texture_dimension: u32,
    /// Browser support level
    browser_support: String,
}

#[wasm_bindgen]
impl WebGPUInfo {
    /// Get GPU vendor name
    #[wasm_bindgen(getter)]
    pub fn vendor(&self) -> String {
        self.vendor.clone()
    }

    /// Get GPU architecture
    #[wasm_bindgen(getter)]
    pub fn architecture(&self) -> String {
        self.architecture.clone()
    }

    /// Get browser support level
    #[wasm_bindgen(getter, js_name = "browserSupport")]
    pub fn browser_support(&self) -> String {
        self.browser_support.clone()
    }

    /// Get a human-readable summary
    #[wasm_bindgen(js_name = "getSummary")]
    pub fn get_summary(&self) -> String {
        if self.available {
            format!(
                "✅ WebGPU Available\n\
                 GPU: {} ({})\n\
                 Max Buffer: {} MB\n\
                 Max Texture: {}px\n\
                 Browser: {}",
                self.vendor,
                self.architecture,
                self.max_buffer_size / 1_048_576, // Convert to MB
                self.max_texture_dimension,
                self.browser_support
            )
        } else {
            format!("❌ WebGPU Not Available\nBrowser: {}", self.browser_support)
        }
    }
}

/// Check WebGPU availability with detailed information
#[wasm_bindgen(js_name = "checkWebGPUSupport")]
pub async fn check_webgpu_support() -> Result<WebGPUInfo, JsValue> {
    console::log_1(&"🔍 Checking WebGPU support...".into());

    let window = window().ok_or_else(|| JsValue::from_str("No window object"))?;
    let navigator = window.navigator();

    // Check if GPU object exists
    let gpu = js_sys::Reflect::get(&navigator, &"gpu".into())?;

    if gpu.is_undefined() {
        console::warn_1(&"⚠️ WebGPU API not found in navigator".into());
        return Ok(WebGPUInfo {
            available: false,
            vendor: "unknown".to_string(),
            architecture: "unknown".to_string(),
            max_buffer_size: 0,
            max_texture_dimension: 0,
            browser_support: detect_browser(),
        });
    }

    // Try to request adapter
    let request_adapter = js_sys::Reflect::get(&gpu, &"requestAdapter".into())?;

    if request_adapter.is_undefined() {
        console::warn_1(&"⚠️ requestAdapter not available".into());
        return Ok(WebGPUInfo {
            available: false,
            vendor: "unknown".to_string(),
            architecture: "unknown".to_string(),
            max_buffer_size: 0,
            max_texture_dimension: 0,
            browser_support: detect_browser(),
        });
    }

    // Call requestAdapter
    let adapter_promise = js_sys::Reflect::apply(
        &request_adapter.dyn_into::<js_sys::Function>()?,
        &gpu,
        &js_sys::Array::new(),
    )?;

    let adapter = JsFuture::from(js_sys::Promise::from(adapter_promise)).await?;

    if adapter.is_null() {
        console::warn_1(&"⚠️ No WebGPU adapter available".into());
        return Ok(WebGPUInfo {
            available: false,
            vendor: "unknown".to_string(),
            architecture: "unknown".to_string(),
            max_buffer_size: 0,
            max_texture_dimension: 0,
            browser_support: detect_browser(),
        });
    }

    // Extract adapter info
    let info = extract_adapter_info(&adapter)?;

    console::log_1(&format!("✅ WebGPU available: {}", info.get_summary()).into());

    Ok(info)
}

/// Extract information from WebGPU adapter
fn extract_adapter_info(adapter: &JsValue) -> Result<WebGPUInfo, JsValue> {
    // Get adapter info
    let info_obj = js_sys::Reflect::get(adapter, &"info".into())?;

    let vendor = if !info_obj.is_undefined() {
        js_sys::Reflect::get(&info_obj, &"vendor".into())?
            .as_string()
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        "unknown".to_string()
    };

    let architecture = if !info_obj.is_undefined() {
        js_sys::Reflect::get(&info_obj, &"architecture".into())?
            .as_string()
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        "unknown".to_string()
    };

    // Get limits
    let limits_obj = js_sys::Reflect::get(adapter, &"limits".into())?;

    let max_buffer_size = if !limits_obj.is_undefined() {
        js_sys::Reflect::get(&limits_obj, &"maxBufferSize".into())?
            .as_f64()
            .unwrap_or(0.0) as u64
    } else {
        268_435_456 // 256 MB default
    };

    let max_texture_dimension = if !limits_obj.is_undefined() {
        js_sys::Reflect::get(&limits_obj, &"maxTextureDimension2D".into())?
            .as_f64()
            .unwrap_or(8192.0) as u32
    } else {
        8192
    };

    Ok(WebGPUInfo {
        available: true,
        vendor,
        architecture,
        max_buffer_size,
        max_texture_dimension,
        browser_support: detect_browser(),
    })
}

/// Detect browser and version
fn detect_browser() -> String {
    let window = match window() {
        Some(w) => w,
        None => return "unknown".to_string(),
    };

    let user_agent = match window.navigator().user_agent() {
        Ok(ua) => ua,
        Err(_) => return "unknown".to_string(),
    };

    if user_agent.contains("Chrome") && !user_agent.contains("Edg") {
        let version = extract_browser_version(&user_agent, "Chrome/");
        format!("Chrome {}", version)
    } else if user_agent.contains("Edg") {
        let version = extract_browser_version(&user_agent, "Edg/");
        format!("Edge {}", version)
    } else if user_agent.contains("Firefox") {
        let version = extract_browser_version(&user_agent, "Firefox/");
        format!("Firefox {}", version)
    } else if user_agent.contains("Safari") && !user_agent.contains("Chrome") {
        let version = extract_browser_version(&user_agent, "Version/");
        format!("Safari {}", version)
    } else {
        "Unknown Browser".to_string()
    }
}

/// Extract version number from user agent string
fn extract_browser_version(user_agent: &str, pattern: &str) -> String {
    if let Some(start) = user_agent.find(pattern) {
        let version_start = start + pattern.len();
        let version_str = &user_agent[version_start..];

        // Take until first space or non-digit/dot character
        version_str
            .split(|c: char| c.is_whitespace() || (!c.is_numeric() && c != '.'))
            .next()
            .unwrap_or("unknown")
            .to_string()
    } else {
        "unknown".to_string()
    }
}

/// Quick WebGPU availability check (returns bool)
#[wasm_bindgen(js_name = "isWebGPUAvailable")]
pub fn is_webgpu_available() -> bool {
    if let Some(window) = window() {
        let navigator = window.navigator();
        if let Ok(gpu) = js_sys::Reflect::get(&navigator, &"gpu".into()) {
            return !gpu.is_undefined();
        }
    }
    false
}

/// Get recommended ML backend based on WebGPU support
#[wasm_bindgen(js_name = "getRecommendedBackend")]
pub async fn get_recommended_backend() -> String {
    match check_webgpu_support().await {
        Ok(info) if info.available => {
            console::log_1(&"💡 Recommendation: Use Burn + wgpu or WebLLM for GPU acceleration".into());
            "webgpu".to_string()
        }
        _ => {
            console::log_1(&"💡 Recommendation: Use Candle CPU (fallback)".into());
            "cpu".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_is_webgpu_available() {
        let available = is_webgpu_available();
        console::log_1(&format!("WebGPU available: {}", available).into());
    }

    #[wasm_bindgen_test]
    async fn test_check_webgpu_support() {
        match check_webgpu_support().await {
            Ok(info) => {
                console::log_1(&info.get_summary().into());
            }
            Err(e) => {
                console::error_1(&format!("Error: {:?}", e).into());
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_detect_browser() {
        let browser = detect_browser();
        console::log_1(&format!("Detected browser: {}", browser).into());
        assert!(!browser.is_empty());
    }
}
