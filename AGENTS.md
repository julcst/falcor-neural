# Falcor + Slang Workspace Instructions

## Scope

- Apply these instructions to the whole workspace unless a task explicitly narrows the scope.
- Treat [docs/development/coding-conventions.md](docs/development/coding-conventions.md) and [docs/development/error-handling.md](docs/development/error-handling.md) as the source of truth for style and error handling.
- Check the repo memory notes in /memories/repo/ for verified workspace-specific caveats before changing shader-heavy code.

## Documentation

- Slang docs: [llms.txt](https://shader-slang.org/docs/llms.txt) and [llms_full.txt](https://shader-slang.org/docs/llms_full.txt).
- For render passes, also check [docs/usage/render-passes.md](docs/usage/render-passes.md).

## Working Approach

- Start from the closest failing behavior, file, symbol, or test.
- Before the first edit, form one local hypothesis and one cheap check that could disprove it.
- Prefer the smallest edit that can test or fix the hypothesis.
- Do not clean up unrelated code while touching a file.
- After the first substantive edit, run the narrowest meaningful validation before broadening scope.
- If a fix fails validation, repair the same slice first before expanding the investigation.

## C++ And Falcor Conventions

- Follow the repository naming, include, and type rules in the docs.
- Use `override` on virtual overrides, `nullptr` instead of `NULL` or `0`, `enum class` instead of `enum`, and `static constexpr` for class constants.
- Prefer `FALCOR_ASSERT` for impossible internal states, `FALCOR_CHECK` or exceptions for runtime and public API errors, and `logWarning` only for non-critical surprises.
- Use `using` aliases instead of `typedef`.
- Do not use `using namespace std`; keep namespace usage local to source files only.
- Preserve the surrounding style when editing existing code.

## Slang And Render Passes

- Keep Slang files next to the host code and use the repository suffix conventions: `.slang`, `.slangh`, `.cs.slang`, `.rt.slang`, `.3d.slang`, or a single-stage suffix when appropriate.
- List every shader file in the relevant `CMakeLists.txt`, including helper `.slang` and `.slangh` files, so deployment works.
- Treat shader edits as runtime-sensitive: a successful build is not enough; run the affected app or script to catch shader compilation errors.
- For buffer-size-dependent passes, implement `compile(...)` and use connected resource sizes to decide when to recompile.
- For shared host/device data, keep the struct in a shared `.slangh` file included by both sides.
- Prefer standalone helper functions over advanced Slang extension patterns unless the exact compiler support has been verified in this workspace.

### Compact Examples

```cpp
// Minimal render-pass shape: create passes, bind resources, execute.
void SlangMLP::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
	createPasses();
	auto pInput = renderData.getTexture(kInput);
	auto pOutput = renderData.getTexture(kOutput);
	FALCOR_ASSERT(pInput && pOutput);
	// Bind resources and run compute passes here.
}
```

```cpp
// Declare properties and persistent buffers near the pass definition.
const std::string kTrainSteps = "trainSteps";
const std::string kBatchSize = "batchSize";
const std::string kLearningRate = "learningRate";

Properties SlangMLP::getProperties() const
{
	Properties props;
	props[kTrainSteps] = mTrainSteps;
	props[kBatchSize] = mBatchSize;
	props[kLearningRate] = mLearningRate;
	return props;
}
```

```cpp
// Buffer-size-dependent passes should use compile() to react to connected sizes.
void AccumulatePhotonsRTX::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
	auto queryCount = compileData.connectedResources.getField(kQueryBuffer)->getWidth() / sizeof(Query);
	if (mQueryCount != queryCount) { mQueryCount = queryCount; FALCOR_CHECK(false, "Recompile with new buffer sizes"); }
}
```


```cpp
// Shared host/device structs should live in a .slangh included by both sides.
struct NRCInput
{
	float3 position;
	float roughness;
#ifndef HOST_CODE
	float3 weight() { return float3(1); }
#endif
};
```

To build a render graph, Python scripts are used like this:

```python
from falcor import *

def render_graph_PT():
	g = RenderGraph("PT")
	g.createPass("accum", "AccumulatePass", {"precisionMode": "Double"})
	g.createPass("pt", "PathTracer", {"maxDiffuseBounces": 8, "maxSpecularBounces": 8})
	g.createPass("vbuff", "VBufferRT", {})
	g.addEdge("vbuff.vbuffer", "pt.vbuffer")
	g.addEdge("pt.color", "accum.input")
	g.markOutput("accum.output")
	return g

m.loadScene("../scenes/cornell_box_bunny.pyscene")
m.addGraph(render_graph_PT())
```

To inspect frames, you can add frames to the frameCapture:

```python
m.frameCapture.addFrames(m.activeGraph, range(0, 1000, 100))
```

To export timings:

```python
m.timingCapture.captureFrameTime("timings.csv")
```

Or for in depth profiling:

```python
m.profiler.enabled = True
m.profiler.start_capture()
for frame in range(256):
    m.renderFrame()
capture = m.profiler.end_capture()
m.profiler.enabled = False

print(f"Mean frame time: {capture['events']['/onFrameRender/RenderGraphExe::execute()/SlangMLP/SlangMLP/gpu_time']['stats']['mean']} ms")
```

## Validation

- Use CMake Tools builds for C++ and shader deployment changes.
- For shader work, validate by running the relevant executable or Mogwai script after building.
- Use the narrowest check that can confirm the touched slice before expanding scope.

## Build And Run

- Use CMake only; the workspace expects shader files to be copied through the CMake build process.
- Prefer the VS Code CMake Tools workflow for configure/build/debug.
- For this repo, a typical shader-pass run looks like `build/linux-gcc/bin/Debug/Mogwai --script scripts/SlangMLP.py`.
- Shader compilation errors often appear at runtime, so run the app after shader changes even if the build succeeds.
- Mogwai is a GUI app; it may need manual termination. Prefer `--headless` for checking shader compilation, but still stop it explicitly when done (e.g., with Ctrl+\\).

## Communication

- Keep progress updates short and factual.
- If requirements are ambiguous, ask only the minimum needed question.
- If something is blocked by tooling or repo constraints, say what is blocked and the smallest viable alternative.
