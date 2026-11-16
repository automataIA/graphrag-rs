# Qwen model integration

Qwen models can be used by applications that provide a compatible browser or
local inference integration. This is outside the zero-configuration native
quickstart. Select a model variant according to the target device, available
memory, context requirements, model license, and provider format.

For browser use, follow the WebLLM/MLC model documentation and ensure WebGPU is
available. Download and initialization time depend on the model and device;
the application should expose a local fallback or a clear error when the model
cannot be loaded.
