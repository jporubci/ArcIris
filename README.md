# ArcIris

|||
|-|-|
| Author | Jozef Porubcin |

## Acknowledgments

This code is based on [IrisTripletMining](https://github.com/Siamul/IrisTripletMining) by [Siamul](https://github.com/Siamul).

[The ArcFace implementation](arcface_torch/losses.py) is from [the official arcface_torch repository](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).

## Setup

[Python 3.9.19](https://www.python.org/downloads/release/python-3919/) was used.

1. Create a Python environment:

   ```bash
   python3 -m venv .venv
   ```

1. Activate it:

   ```bash
   source .venv/bin/activate
   ```

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running Locally

```bash
python main.py
```

## Running on CRC

1. Submit the job:

   ```bash
   qsub train.sh -notify
   ```

1. Check your jobs:

   ```bash
   qstat -u $USER
   ```

## Future Work

### Class Activation Mapping

1. Extract embeddings
1. Train classifier on subset
1. Get gradients
