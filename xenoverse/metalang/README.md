# Introduction

Generating Randomized Pseudo Meta-Language for Benchmarking Long-Term Dependency In-Context Learning

- MetaLangv1: Generate by repeated random sequences
- MetaLangv2: Generate from randomized n-gram neural network models

# Usage

### PYTHON APIs

```python
import gym
import xenoverse.metalang
from xenoverse.metalang import TaskSamplerV2

# Initialize the generator
generator = gym.make("meta-language-v2")

# Sample a task
task = TaskSamplerV2(n_gram=3,
                    n_embedding=16,
                    _lambda=5.0,
                    ...)

# Set the task
generator.set_task(task)

# Generate sequences from the task
batch_sample = generator.batch_generator(batch_size=8)
```

### COMMAND LINES

```bash
# Sample 100 tasks first
python -m xenoverse.metalang.generator --sample_type tasks --samples 100 --output tasks.pkl ...

# Sample 1000 sequences from the 100 tasks
python -m xenoverse.metalang.generator --sample_type sequences --task_file tasks.pkl --samples 1000 --output sequences.txt --output_type txt ...

# Or generate 1000 sequences by randomly sample tasks on the fly
python -m xenoverse.metalang.generator --sample_type sequences --samples 1000 --output sequences.txt --output_type txt ...
```

#### show all the options

```bash
python -m xenoverse.metalang.generator --help 

Generating Meta Language Tasks or Sequences

optional arguments:
  -h, --help            show this help message and exit

  --version {v1,v2}

  --sample_type {tasks,sequences}
                        Generate tasks or sequences

  --task_file TASK_FILE
                        Specify task file to generate from if the sample_type is sequences. Default will generate task on the fly.

  --vocab_size VOCAB_SIZE

  --embedding_size EMBEDDING_SIZE

  --hidden_size HIDDEN_SIZE

  --patterns_number PATTERNS_NUMBER

  --error_rate ERROR_RATE

  --n_gram N_GRAM

  --lambda_weight LAMBDA_WEIGHT
                        Lambda weight multiplied for softmax sampling in MetaLangV2

  --batch_size BATCH_SIZE

  --sequence_length SEQUENCE_LENGTH

  --samples SAMPLES     number of sequences / tasks to generate

  --output_type {txt,npy}

  --output OUTPUT
```

