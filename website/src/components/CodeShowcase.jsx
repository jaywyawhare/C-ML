import { useState, useCallback, useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const tabs = [
  { id: 'quick', label: 'Quick Start' },
  { id: 'training', label: 'Training' },
  { id: 'dataset', label: 'Dataset Hub' },
  { id: 'gpu', label: 'GPU' },
]

const code = {
  quick: {
    plain: `#include "cml.h"

int main(void) {
    cml_init();

    // Build a simple model
    Sequential* model = cml_nn_sequential();
    DeviceType dev = cml_get_default_device();
    DType dt = cml_get_default_dtype();
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(4, 16, dt, dev, true));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
    model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 3, dt, dev, true));

    // Forward pass
    Tensor* input = tensor_randn((int[]){1, 4}, 2, dt, dev);
    Tensor* output = cml_nn_module_forward((Module*)model, input);

    cml_cleanup();
    return 0;
}`,
    jsx: (
      <>
        <span className="pp">#include</span> <span className="str">"cml.h"</span>{'\n\n'}
        <span className="type">int</span> <span className="fn">main</span>(<span className="type">void</span>) {'{\n'}
        {'    '}<span className="fn">cml_init</span>();{'\n\n'}
        {'    '}<span className="cm">// Build a simple model</span>{'\n'}
        {'    '}<span className="type">Sequential</span>* model = <span className="fn">cml_nn_sequential</span>();{'\n'}
        {'    '}<span className="type">DeviceType</span> dev = <span className="fn">cml_get_default_device</span>();{'\n'}
        {'    '}<span className="type">DType</span> dt = <span className="fn">cml_get_default_dtype</span>();{'\n'}
        {'    '}model = <span className="fn">cml_nn_sequential_add</span>(model, (<span className="type">Module</span>*)<span className="fn">cml_nn_linear</span>(<span className="num">4</span>, <span className="num">16</span>, dt, dev, <span className="kw">true</span>));{'\n'}
        {'    '}model = <span className="fn">cml_nn_sequential_add</span>(model, (<span className="type">Module</span>*)<span className="fn">cml_nn_relu</span>(<span className="kw">false</span>));{'\n'}
        {'    '}model = <span className="fn">cml_nn_sequential_add</span>(model, (<span className="type">Module</span>*)<span className="fn">cml_nn_linear</span>(<span className="num">16</span>, <span className="num">3</span>, dt, dev, <span className="kw">true</span>));{'\n\n'}
        {'    '}<span className="cm">// Forward pass</span>{'\n'}
        {'    '}<span className="type">Tensor</span>* input = <span className="fn">tensor_randn</span>((<span className="type">int</span>[]){'{'}
        <span className="num">1</span>, <span className="num">4</span>{'}'}, <span className="num">2</span>, dt, dev);{'\n'}
        {'    '}<span className="type">Tensor</span>* output = <span className="fn">cml_nn_module_forward</span>((<span className="type">Module</span>*)model, input);{'\n\n'}
        {'    '}<span className="fn">cml_cleanup</span>();{'\n'}
        {'    '}<span className="kw">return</span> <span className="num">0</span>;{'\n'}
        {'}'}
      </>
    ),
  },
  training: {
    plain: `#include "cml.h"

// Load dataset and build model
Dataset* ds = cml_dataset_load("iris");
dataset_normalize(ds, "minmax");
Dataset *train, *test;
dataset_split(ds, 0.8f, &train, &test);

Sequential* model = cml_nn_sequential();
model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(4, 16, dt, dev, true));
model = cml_nn_sequential_add(model, (Module*)cml_nn_relu(false));
model = cml_nn_sequential_add(model, (Module*)cml_nn_linear(16, 3, dt, dev, true));

// Train with Adam optimizer
Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
for (int epoch = 0; epoch < 100; epoch++) {
    cml_optim_zero_grad(opt);
    Tensor* out = cml_nn_module_forward((Module*)model, train->X);
    Tensor* loss = cml_nn_mse_loss(out, train->y);
    cml_backward(loss, NULL, false, false);
    cml_optim_step(opt);
}`,
    jsx: (
      <>
        <span className="pp">#include</span> <span className="str">"cml.h"</span>{'\n\n'}
        <span className="cm">// Load dataset and build model</span>{'\n'}
        <span className="type">Dataset</span>* ds = <span className="fn">cml_dataset_load</span>(<span className="str">"iris"</span>);{'\n'}
        <span className="fn">dataset_normalize</span>(ds, <span className="str">"minmax"</span>);{'\n'}
        <span className="type">Dataset</span> *train, *test;{'\n'}
        <span className="fn">dataset_split</span>(ds, <span className="num">0.8f</span>, &train, &test);{'\n\n'}
        <span className="type">Sequential</span>* model = <span className="fn">cml_nn_sequential</span>();{'\n'}
        model = <span className="fn">cml_nn_sequential_add</span>(model, (<span className="type">Module</span>*)<span className="fn">cml_nn_linear</span>(<span className="num">4</span>, <span className="num">16</span>, dt, dev, <span className="kw">true</span>));{'\n'}
        model = <span className="fn">cml_nn_sequential_add</span>(model, (<span className="type">Module</span>*)<span className="fn">cml_nn_relu</span>(<span className="kw">false</span>));{'\n'}
        model = <span className="fn">cml_nn_sequential_add</span>(model, (<span className="type">Module</span>*)<span className="fn">cml_nn_linear</span>(<span className="num">16</span>, <span className="num">3</span>, dt, dev, <span className="kw">true</span>));{'\n\n'}
        <span className="cm">// Train with Adam optimizer</span>{'\n'}
        <span className="type">Optimizer</span>* opt = <span className="fn">cml_optim_adam_for_model</span>((<span className="type">Module</span>*)model, <span className="num">0.01f</span>, <span className="num">0.0f</span>, <span className="num">0.9f</span>, <span className="num">0.999f</span>, <span className="num">1e-8f</span>);{'\n'}
        <span className="kw">for</span> (<span className="type">int</span> epoch = <span className="num">0</span>; epoch {'<'} <span className="num">100</span>; epoch++) {'{\n'}
        {'    '}<span className="fn">cml_optim_zero_grad</span>(opt);{'\n'}
        {'    '}<span className="type">Tensor</span>* out = <span className="fn">cml_nn_module_forward</span>((<span className="type">Module</span>*)model, train{'->'}X);{'\n'}
        {'    '}<span className="type">Tensor</span>* loss = <span className="fn">cml_nn_mse_loss</span>(out, train{'->'}y);{'\n'}
        {'    '}<span className="fn">cml_backward</span>(loss, <span className="kw">NULL</span>, <span className="kw">false</span>, <span className="kw">false</span>);{'\n'}
        {'    '}<span className="fn">cml_optim_step</span>(opt);{'\n'}
        {'}'}
      </>
    ),
  },
  dataset: {
    plain: `#include "cml.h"

// One-liner dataset loading with auto-download and caching
Dataset* iris     = cml_dataset_load("iris");       // 150 samples, 4 features
Dataset* mnist    = cml_dataset_load("mnist");      // 70k samples, 784 features
Dataset* cifar    = cml_dataset_load("cifar10");    // 60k samples, 3072 features
Dataset* custom   = cml_dataset_from_csv("data.csv", -1);  // Custom CSV

// Built-in preprocessing
dataset_normalize(iris, "minmax");
dataset_shuffle(iris, 42);

Dataset *train, *test;
dataset_split(iris, 0.8f, &train, &test);

// Available: iris, wine, breast_cancer, boston, mnist,
//            fashion_mnist, cifar10, airline, digits`,
    jsx: (
      <>
        <span className="pp">#include</span> <span className="str">"cml.h"</span>{'\n\n'}
        <span className="cm">// One-liner dataset loading with auto-download and caching</span>{'\n'}
        <span className="type">Dataset</span>* iris{'     '}= <span className="fn">cml_dataset_load</span>(<span className="str">"iris"</span>);{'       '}<span className="cm">// 150 samples, 4 features</span>{'\n'}
        <span className="type">Dataset</span>* mnist{'    '}= <span className="fn">cml_dataset_load</span>(<span className="str">"mnist"</span>);{'      '}<span className="cm">// 70k samples, 784 features</span>{'\n'}
        <span className="type">Dataset</span>* cifar{'    '}= <span className="fn">cml_dataset_load</span>(<span className="str">"cifar10"</span>);{'    '}<span className="cm">// 60k samples, 3072 features</span>{'\n'}
        <span className="type">Dataset</span>* custom{'   '}= <span className="fn">cml_dataset_from_csv</span>(<span className="str">"data.csv"</span>, <span className="num">-1</span>);{'  '}<span className="cm">// Custom CSV</span>{'\n\n'}
        <span className="cm">// Built-in preprocessing</span>{'\n'}
        <span className="fn">dataset_normalize</span>(iris, <span className="str">"minmax"</span>);{'\n'}
        <span className="fn">dataset_shuffle</span>(iris, <span className="num">42</span>);{'\n\n'}
        <span className="type">Dataset</span> *train, *test;{'\n'}
        <span className="fn">dataset_split</span>(iris, <span className="num">0.8f</span>, &train, &test);{'\n\n'}
        <span className="cm">// Available: iris, wine, breast_cancer, boston, mnist,</span>{'\n'}
        <span className="cm">//            fashion_mnist, cifar10, airline, digits</span>
      </>
    ),
  },
  gpu: {
    plain: `#include "cml.h"

// Automatic GPU dispatch: CUDA -> ROCm -> Vulkan -> Metal -> WebGPU -> OpenCL -> CPU
DeviceType dev = cml_get_default_device();  // Auto-detects best backend

// Create tensors on GPU
Tensor* a = tensor_randn((int[]){1024, 1024}, 2, DTYPE_FLOAT32, dev);
Tensor* b = tensor_randn((int[]){1024, 1024}, 2, DTYPE_FLOAT32, dev);

// Operations automatically dispatched to GPU
Tensor* c = tensor_matmul(a, b);

// IR graph optimization with kernel fusion
IRGraph* graph = ir_graph_create();
ir_graph_add_op(graph, IR_MATMUL, a, b);
ir_graph_optimize(graph);  // Fuses ops, eliminates redundancy
ir_graph_execute(graph);`,
    jsx: (
      <>
        <span className="pp">#include</span> <span className="str">"cml.h"</span>{'\n\n'}
        <span className="cm">{'// Automatic GPU dispatch: CUDA -> ROCm -> Vulkan -> Metal -> WebGPU -> OpenCL -> CPU'}</span>{'\n'}
        <span className="type">DeviceType</span> dev = <span className="fn">cml_get_default_device</span>();{'  '}<span className="cm">{'// Auto-detects best backend'}</span>{'\n\n'}
        <span className="cm">// Create tensors on GPU</span>{'\n'}
        <span className="type">Tensor</span>* a = <span className="fn">tensor_randn</span>((<span className="type">int</span>[]){'{'}
        <span className="num">1024</span>, <span className="num">1024</span>{'}'}, <span className="num">2</span>, DTYPE_FLOAT32, dev);{'\n'}
        <span className="type">Tensor</span>* b = <span className="fn">tensor_randn</span>((<span className="type">int</span>[]){'{'}
        <span className="num">1024</span>, <span className="num">1024</span>{'}'}, <span className="num">2</span>, DTYPE_FLOAT32, dev);{'\n\n'}
        <span className="cm">// Operations automatically dispatched to GPU</span>{'\n'}
        <span className="type">Tensor</span>* c = <span className="fn">tensor_matmul</span>(a, b);{'\n\n'}
        <span className="cm">// IR graph optimization with kernel fusion</span>{'\n'}
        <span className="type">IRGraph</span>* graph = <span className="fn">ir_graph_create</span>();{'\n'}
        <span className="fn">ir_graph_add_op</span>(graph, IR_MATMUL, a, b);{'\n'}
        <span className="fn">ir_graph_optimize</span>(graph);{'  '}<span className="cm">// Fuses ops, eliminates redundancy</span>{'\n'}
        <span className="fn">ir_graph_execute</span>(graph);
      </>
    ),
  },
}

export default function CodeShowcase() {
  const [tab, setTab] = useState('quick')
  const [copyLabel, setCopyLabel] = useState('copy')
  const ref = useRef(null)

  const onCopy = useCallback(() => {
    navigator.clipboard.writeText(code[tab].plain)
    setCopyLabel('copied')
    setTimeout(() => setCopyLabel('copy'), 1200)
  }, [tab])

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current,
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 80%' } }
      )
    })
    return () => ctx.revert()
  }, [])

  return (
    <section id="code" ref={ref} style={{ opacity: 0 }}>
      <div className="section-center" style={{ marginBottom: 48 }}>
        <div className="section-label">Developer experience</div>
        <h2 className="section-heading">10 lines to first training loop.</h2>
        <p className="section-sub center">Simple C API. Full control when you need it.</p>
      </div>

      <div className="code-wrap">
        <div className="code-tabs">
          {tabs.map(t => (
            <button
              key={t.id}
              className={`code-tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => { setTab(t.id); setCopyLabel('copy') }}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="code-body">
          <button className="code-copy" onClick={onCopy}>{copyLabel}</button>
          <AnimatePresence mode="wait">
            <motion.pre
              key={tab}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              {code[tab].jsx}
            </motion.pre>
          </AnimatePresence>
        </div>
      </div>
    </section>
  )
}
