# Visual Puzzles

## Benchmark Data Preparation

Download the benchmark data first, see the main [README.md](../README.md) for instructions. Then, copy the data to the `visual_puzzles/data` directory:

```bash
# under the Thinking-with-Video root directory
cp -r VideoThinkBench/Vision-Centric_Reasoning/visual_puzzles/* visual_puzzles/data
```

## Test Sora-2

```bash
bash scripts/run.sh
```

## Generate New Data

```bash
bash scripts/gen.sh
```