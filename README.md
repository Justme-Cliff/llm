# Mini LLM — A Language Model From Scratch in C

> A fully hand-written GRU language model that runs entirely on your CPU.
> No Python. No PyTorch. No CUDA. Just C, a CSV, and your machine.

---

## What This Is

Mini LLM is a complete language model pipeline built from scratch in C11. It implements every layer of the stack yourself — tokenization, model architecture, forward pass, backpropagation, training loop, checkpoint saving, loss charting, and a local web chat interface.

It is not trying to compete with GPT-4. It is designed to show you exactly how a language model works under the hood, with no magic hidden behind a framework.

---

## Demo

```
User: What is a neural network?
Assistant: A neural network is a model inspired by the brain that learns patterns from data.

User: What is backpropagation?
Assistant: Backpropagation computes gradients through a network so weights can be updated.
```

```
pretrain [########################--------]  75% | step 4200 | avg loss 3.1042 | 450.2s/600s
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MINI LLM PIPELINE                        │
├─────────────┬──────────────┬───────────────┬────────────────────┤
│   Dataset   │  Tokenizer   │   GRU Model   │   Chat Server      │
│             │              │               │                    │
│  data.csv   │  Byte-level  │  Embeddings   │  HTTP on :8080     │
│  779 Q&A    │  256 tokens  │  GRU layers   │  Serves UI         │
│  pairs      │  vocab=256   │  hidden=128   │  POST /api/chat    │
└─────────────┴──────────────┴───────────────┴────────────────────┘
```

```
TEXT INPUT
    │
    ▼
┌─────────────────┐
│  Byte Tokenizer │  "Hello" → [72, 101, 108, 108, 111]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  token_id → float[128]
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────┐
│   GRU Cell ×T   │◄──────│  h_prev  │  hidden state carried forward
│                 │       └──────────┘
│  reset gate r   │
│  update gate z  │
│  new gate n     │
└────────┬────────┘
         │ h_next
         ▼
┌─────────────────┐
│  Output Layer   │  h → logits[256]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Softmax      │  logits → probabilities[256]
└────────┬────────┘
         │
         ▼
     next token
```

---

## Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                     TRAINING FLOW                            │
│                                                              │
│  1. Load data.csv                                            │
│  2. Tokenize entire dataset into token stream                │
│  3. Sample random windows of length seq_len                  │
│                                                              │
│  ┌─── STAGE 1: PRETRAIN ───────────────────────────────┐    │
│  │  • Broad random window sampling                      │    │
│  │  • Learning rate: 3e-4                               │    │
│  │  • Runs for N seconds (default: 3600)                │    │
│  │  • Saves pretrain.ckpt + pretrain_loss.png           │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─── STAGE 2: FINETUNE ──────────────────────────────┐     │
│  │  • 70% samples biased toward "Assistant:" sections  │     │
│  │  • Lower learning rate: 1e-4                        │     │
│  │  • Runs for N seconds (default: 1800)               │     │
│  │  • Saves finetune.ckpt + finetune_loss.png          │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

The model learns by **next-token prediction**: given the previous tokens, predict the next one. The loss is cross-entropy between the predicted probability distribution and the correct token. Gradients flow back through time (BPTT) to update all weights.

---

## GRU Cell — How It Works

A GRU (Gated Recurrent Unit) processes one token at a time and maintains a hidden state that carries information across the sequence.

```
At each timestep t:

  Input:   x_t  (embedding of current token)
           h_{t-1}  (hidden state from previous step)

  Gates:
    reset gate:   r = sigmoid(Wr·x_t + Ur·h_{t-1} + br)
    update gate:  z = sigmoid(Wz·x_t + Uz·h_{t-1} + bz)
    new gate:     n = tanh(Wn·x_t + Un·(r ⊙ h_{t-1}) + bn)

  Output:
    h_t = (1 - z) ⊙ n  +  z ⊙ h_{t-1}

  ⊙ = element-wise multiply
```

- The **reset gate** controls how much past state to forget
- The **update gate** controls how much new vs old state to keep
- The **new gate** computes the candidate new state

This is implemented entirely by hand in `src/model_gru.c` — no external libraries.

---

## File Structure

```
mini-llm/
│
├── src/
│   ├── train_main.c          # Training entry point — pretrain + finetune loop
│   ├── chat_server_main.c    # HTTP chat server — serves UI + /api/chat endpoint
│   ├── model_gru.h / .c      # GRU model — weights, forward pass, backprop, save/load
│   ├── tokenizer_byte.h / .c # Byte tokenizer — 256 token vocab, text ↔ token IDs
│   ├── dataset_csv.h / .c    # CSV loader — parses data.csv into token stream
│   ├── plot_png.h / .c       # PNG chart generator — renders loss curves
│   ├── plot_svg.h / .c       # SVG chart generator (alternative)
│   └── util.h / .c           # Utilities — timing, file I/O, RNG, mkdir
│
├── ui/
│   ├── index.html            # Chat web interface
│   └── app.js                # Frontend — sends prompts, renders responses
│
├── data.csv                  # Training dataset (~779 Q&A pairs)
├── CMakeLists.txt            # Build configuration (CMake + Ninja)
├── run.sh                    # One-command build + train + serve script
└── README.md
```

---

## Dataset Format

The dataset is a CSV with a single `text` column. Each row is one dialogue exchange:

```csv
text
"User: What is a CPU?\nAssistant: A CPU is a processor that executes instructions in a computer."
"User: What is RAM?\nAssistant: RAM is temporary fast memory used by the computer for active tasks."
```

- Header must be `text`
- Each row is a quoted string
- `\n` separates User and Assistant turns
- The model uses `Assistant:` as a marker during fine-tuning to bias sampling

The current dataset has **779 rows** covering: math, science, programming, ML/AI, history, geography, health, music, sports, cooking, finance, physics, space, and more.

---

## Requirements

| Tool    | Version  | Purpose                     |
|---------|----------|-----------------------------|
| GCC     | any      | C compiler (MinGW on Windows)|
| CMake   | ≥ 3.20   | Build system                |
| Ninja   | any      | Fast build backend          |

On Windows install via [MSYS2](https://www.msys2.org/) or [winlibs](https://winlibs.com/).

---

## Quick Start

### One command (recommended)

```bash
bash run.sh
```

This will:
1. Build the project with CMake + Ninja
2. Train for 1 hour pretrain + 30 min finetune
3. Start the chat server at `http://localhost:8080`

### Manual steps

**Build:**
```bash
cmake -G "Ninja" -S . -B build_ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_ninja --config Release
```

**Train:**
```bash
./build_ninja/train_main.exe \
  --data data.csv \
  --out out \
  --pretrain-seconds 3600 \
  --finetune-seconds 1800
```

**Chat:**
```bash
./build_ninja/chat_server_main.exe --ckpt out/finetune.ckpt --port 8080 --ui ui
```

Then open `http://localhost:8080`.

---

## All Training Flags

| Flag                  | Default     | Description                                        |
|-----------------------|-------------|----------------------------------------------------|
| `--data`              | `data.csv`  | Path to training CSV                               |
| `--out`               | `out`       | Output directory for checkpoints and charts        |
| `--pretrain-seconds`  | `3600`      | How long to run the pretrain stage                 |
| `--finetune-seconds`  | `1800`      | How long to run the finetune stage                 |
| `--hidden`            | `128`       | GRU hidden size (bigger = smarter but slower)      |
| `--seq-len`           | `96`        | Sequence length per training window                |
| `--lr-pretrain`       | `3e-4`      | Learning rate for pretrain stage                   |
| `--lr-finetune`       | `1e-4`      | Learning rate for finetune stage                   |
| `--batch`             | `1`         | Batch size                                         |
| `--log-every`         | `20`        | Log to CSV every N steps                           |
| `--grad-clip`         | `1.0`       | Gradient clipping threshold                        |
| `--assistant-bias`    | `0.7`       | Probability of sampling near "Assistant:" in finetune |

**Chat server flags:**

| Flag      | Default          | Description               |
|-----------|------------------|---------------------------|
| `--ckpt`  | *(required)*     | Path to `.ckpt` checkpoint |
| `--port`  | `8080`           | Port to listen on         |
| `--ui`    | `ui`             | Path to UI directory      |

---

## Output Files

After training the output directory contains:

```
out/
├── pretrain.ckpt        # Saved model weights after pretrain
├── finetune.ckpt        # Saved model weights after finetune (use this for chat)
├── pretrain_loss.csv    # Loss log for pretrain stage (step, loss, elapsed)
├── finetune_loss.csv    # Loss log for finetune stage
├── pretrain_loss.png    # Loss curve chart for pretrain
└── finetune_loss.png    # Loss curve chart for finetune
```

The `.ckpt` files are raw binary dumps of the model weight arrays. They include:
- Embedding matrix `[vocab_size × hidden_size]`
- GRU weight matrices: Wz, Uz, bz, Wr, Ur, br, Wn, Un, bn
- Output projection: out_W `[vocab_size × hidden_size]`, out_b `[vocab_size]`

---

## Chat Interface

The UI is a local web app served by the C HTTP server.

```
┌──────────────────────────────────────────┐
│  ┌────────┐  Mini LLM  [BETA]            │
│  │  M     │                              │
│  └────────┘  New chat                   │
│  ─────────────────────────────────────  │
│  Model: GRU · CPU · Local               │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │           Mini LLM 🤖            │   │
│  │    A small model on your CPU.    │   │
│  └──────────────────────────────────┘   │
│                                          │
│  You:  What is machine learning?         │
│                                          │
│  AI:   Machine learning is a method      │
│        where computers learn from data.  │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  Message Mini LLM…          [▶]  │   │
│  └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

- Press **Enter** to send, **Shift+Enter** for new line
- **New chat** button clears the conversation
- Adjust **Max tokens** to control response length
- The server handles one request at a time (single-threaded)

### How the Chat API Works

```
Browser                    C Server                    Model
   │                           │                          │
   │  POST /api/chat           │                          │
   │  { prompt, max_tokens }   │                          │
   │──────────────────────────►│                          │
   │                           │  Build prompt template:  │
   │                           │  "User: {input}\n        │
   │                           │   Assistant: "           │
   │                           │─────────────────────────►│
   │                           │                          │ tokenize
   │                           │                          │ run GRU forward
   │                           │                          │ sample next token
   │                           │                          │ repeat × max_tokens
   │                           │◄─────────────────────────│
   │                           │  generated text          │
   │◄──────────────────────────│                          │
   │  { response: "..." }      │                          │
```

---

## How Backpropagation Works Here

The model is trained with **Backpropagation Through Time (BPTT)**. At every update step:

1. **Forward pass** — run all T timesteps, store all intermediate activations (z, r, n, h at every step)
2. **Compute loss** — cross-entropy between predicted probabilities and actual next tokens, summed over all T positions and all B batch elements
3. **Backward pass** — unroll the computation graph in reverse from t=T-1 to t=0, accumulating gradients for every weight matrix
4. **Gradient clipping** — clamp all gradients to `[-grad_clip, +grad_clip]` to prevent exploding gradients
5. **SGD update** — `w = w - lr * grad` for every parameter

All of this is implemented by hand in `train_main.c`. No autograd engine.

---

## Tokenizer

The tokenizer is byte-level with a vocabulary of exactly **256 tokens** — one per possible byte value (0–255). This means:

- No out-of-vocabulary tokens ever
- Works on any language or content without special handling
- Token ID is literally the byte value of the character
- `'A'` = token 65, `' '` = token 32, `'\n'` = token 10

Simple, fast, and zero external dependencies.

---

## Performance Expectations

| Setting           | Steps/sec (approx) | Quality   |
|-------------------|--------------------|-----------|
| hidden=32         | ~800               | Very weak |
| hidden=128        | ~200               | Reasonable|
| hidden=256        | ~60                | Better    |
| hidden=512        | ~15                | Slow      |

These numbers vary significantly by CPU. The default `hidden=128` is a good balance of speed and quality for a 1–2 hour training run.

The model will produce **coherent short responses** on topics covered in the training data. It will not generalize well outside the dataset — that is expected for a model this size.

---

## Troubleshooting

**Build fails: `cmake` not found**
Install CMake and make sure it is on your PATH.

**Build fails: `ninja` not found**
Install Ninja. With MSYS2: `pacman -S mingw-w64-x86_64-ninja`

**`gcc` errors about C11**
Make sure you have GCC 5+ or any modern MinGW build.

**Training is very slow**
Reduce hidden size: `--hidden 64` or `--hidden 32`

**Chat gives gibberish**
Normal for short training runs. Try training longer or adding more data to `data.csv`.

**Port 8080 already in use**
Use a different port: `--port 9090` and open `http://localhost:9090`

---

## Extending the Project

**Add more training data**
Just add rows to `data.csv` in the format:
```
"User: your question\nAssistant: your answer"
```

**Make the model bigger**
```bash
./build_ninja/train_main.exe --hidden 256 --seq-len 128 ...
```

**Train longer**
```bash
./build_ninja/train_main.exe --pretrain-seconds 7200 --finetune-seconds 3600 ...
```

**Continue from a checkpoint**
Not yet implemented — the model initializes fresh each run.

---

## What This Project Teaches You

By reading the source you will understand:

- How a GRU cell computes its gates and hidden state
- How embeddings map discrete tokens to continuous vectors
- How cross-entropy loss is computed from logits
- How BPTT unrolls a recurrent network in reverse
- How gradient clipping prevents training instability
- How a two-stage pretrain/finetune pipeline works
- How a plain C HTTP server parses requests and serves JSON
- How a PNG is generated from raw pixel data without any library

---

## License

MIT — do whatever you want with it.

---

> Built in C. Runs on CPU. No dependencies. No magic.
