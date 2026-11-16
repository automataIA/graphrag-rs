//! Every shipped configuration template must load through the public loader.

#[test]
fn shipped_templates_load() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let dirs = [
        "config/templates",
        "graphrag-core/templates",
        "graphrag-core/config-examples",
        "tests/e2e/configs",
    ];
    let mut failures = Vec::new();
    let mut checked = 0;
    for dir in dirs {
        for entry in std::fs::read_dir(format!("{root}/{dir}")).unwrap() {
            let path = entry.unwrap().path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext, "toml" | "json5" | "json")
                || (ext == "json5" && !cfg!(feature = "json5-support"))
            {
                continue;
            }
            checked += 1;
            if let Err(e) = graphrag_core::config::loader::load_config(path.to_str().unwrap()) {
                failures.push(format!("{}: {e}", path.display()));
            }
        }
    }
    assert!(checked > 0, "no templates found");
    assert!(
        failures.is_empty(),
        "templates failed to load:\n{}",
        failures.join("\n")
    );
}
