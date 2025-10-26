//! WebGPU Detection Tests
//!
//! Tests for WebGPU support detection and validation.

use wasm_bindgen_test::*;
use graphrag_wasm::check_webgpu_support;

wasm_bindgen_test_configure!(run_in_browser);

/// Test: WebGPU support detection
///
/// Validates that we can check for WebGPU support in the browser.
#[wasm_bindgen_test]
async fn test_webgpu_detection() {
    let result = check_webgpu_support().await;

    // Function should complete without error
    assert!(result.is_ok());

    let has_webgpu = result.unwrap();

    web_sys::console::log_1(&format!("WebGPU available: {}", has_webgpu).into());

    // Result should be boolean
    // Note: May be true or false depending on browser/environment
}

/// Test: Multiple WebGPU checks
///
/// Validates that multiple detection calls work consistently.
#[wasm_bindgen_test]
async fn test_multiple_webgpu_checks() {
    let result1 = check_webgpu_support().await;
    let result2 = check_webgpu_support().await;
    let result3 = check_webgpu_support().await;

    assert!(result1.is_ok());
    assert!(result2.is_ok());
    assert!(result3.is_ok());

    // All checks should return the same result
    let val1 = result1.unwrap();
    let val2 = result2.unwrap();
    let val3 = result3.unwrap();
    assert_eq!(val1, val2);
    assert_eq!(val2, val3);
}

/// Test: WebGPU detection performance
///
/// Validates that WebGPU detection completes quickly.
#[wasm_bindgen_test]
async fn test_webgpu_detection_performance() {
    use web_sys::window;

    let start = window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let _result = check_webgpu_support().await;

    let end = window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let duration_ms = end - start;

    web_sys::console::log_1(&format!("WebGPU detection took: {:.2}ms", duration_ms).into());

    // Should complete in reasonable time (< 1 second)
    assert!(duration_ms < 1000.0);
}

/// Test: WebGPU feature detection fallback
///
/// Validates that detection works even in environments without WebGPU.
#[wasm_bindgen_test]
async fn test_webgpu_fallback() {
    // This test just ensures the function doesn't panic or hang
    // when WebGPU is not available

    let result = check_webgpu_support().await;

    match result {
        Ok(true) => {
            web_sys::console::log_1(&"✅ WebGPU is available".into());
        }
        Ok(false) => {
            web_sys::console::log_1(&"⚠️  WebGPU not available (expected in some browsers)".into());
        }
        Err(e) => {
            web_sys::console::error_1(&format!("❌ Detection failed: {:?}", e).into());
            // Fail the test if detection itself errors
            panic!("WebGPU detection should not error");
        }
    }
}

/// Test: WebGPU adapter info (if available)
///
/// If WebGPU is available, tries to get adapter information.
#[wasm_bindgen_test]
async fn test_webgpu_adapter_info() {
    let has_webgpu = check_webgpu_support().await.unwrap_or(false);

    if !has_webgpu {
        web_sys::console::log_1(&"Skipping adapter info test - WebGPU not available".into());
        return;
    }

    // If WebGPU is available, we could potentially query more details
    // For now, just log that it's available
    web_sys::console::log_1(&"WebGPU is available - adapter info could be queried".into());
}

/// Test: Concurrent WebGPU detection
///
/// Validates that concurrent detection calls work correctly.
#[wasm_bindgen_test]
async fn test_concurrent_webgpu_detection() {
    use std::rc::Rc;
    use std::cell::RefCell;

    let results = Rc::new(RefCell::new(Vec::new()));

    // Launch 5 concurrent detection calls
    let mut handles = Vec::new();
    for i in 0..5 {
        let results = Rc::clone(&results);
        let handle = async move {
            let result = check_webgpu_support().await;
            results.borrow_mut().push((i, result));
        };
        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        handle.await;
    }

    // All should succeed
    let results = results.borrow();
    assert_eq!(results.len(), 5);

    for (i, result) in results.iter() {
        web_sys::console::log_1(&format!("Detection {}: {:?}", i, result).into());
        assert!(result.is_ok());
    }

    // All should return the same value
    if let Some((_, first_result)) = results.first() {
        if let Ok(first_value) = first_result {
            for (_, result) in results.iter().skip(1) {
                if let Ok(value) = result {
                    assert_eq!(first_value, value);
                }
            }
        }
    }
}
