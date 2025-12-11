# PIVOT: Prompting with Iterative Visual Optimization

PIVOT is a visual-prompting framework for zero-shot drone navigation using Vision-Language Models (VLMs). Instead of producing actions directly, PIVOT overlays candidate trajectories on the input image and asks a VLM to pick the safest or most goal-directed option.

Through iterative refinement, the system repeatedly samples around the VLM-selected candidates, narrowing the search space and improving decisions without any robot-specific training.

(See paper: https://arxiv.org/abs/2402.07872)

## Features

- **Visual prompting interface** for VLM-based trajectory selection
- **Iterative candidate refinement** for improved decision quality
- **Zero-shot**: no task-specific data or training
- **AirSim integration** for simulation and evaluation
- Gemini / OpenAI-compatible API support

## Installation

### Prerequisites

- Python 3.13+
- [Microsoft AirSim](https://github.com/microsoft/AirSim) installed and running
- Google Gemini API key or OpenAI-compatible API key

### Setup

1. **Clone and navigate to the repository**:
   ```bash
   git clone git@github.com:ChanJoon/PIVOT.git
   cd PIVOT
   ```

2. **Install dependencies**:
   Make sure uv is installed.
   Sync the project dependencies and activate the virtual environment:
   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. **Configure API keys**:

   The `.env` file is symlinked to `../see-point-fly/.env`. Edit it to add your API keys:
   ```bash
   # Gemini API Configuration
   GEMINI_API_KEY=your_key_here

   # OR OpenAI-compatible API Configuration
   OPENAI_API_KEY=your_key_here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   ```

4. **Configure AirSim camera**:

   AirSim's default camera resolution (256x144) is too low. Copy the settings file:
   ```bash
   cp ./settings.json.example ~/Documents/AirSim/<environment binary>/settings.json
   ```

   Then restart AirSim to apply the new 1920x1080 resolution.

## Usage

### Basic Usage

1. **Start AirSim** with your preferred environment

2. **Run PIVOT**:
   ```bash
   python -m pivot.main
   ```

   Or if installed:
   ```bash
   pivot
   ```

### Command Line Options

```bash
# Use custom configuration
python -m pivot.main --config my_config.yaml

# Override navigation instruction
python -m pivot.main --instruction "fly to the blue car"

# Run single navigation then exit (for testing)
python -m pivot.main --single-shot

# Enable debug mode
python -m pivot.main --debug
```

## Visualization Output

PIVOT saves visualizations to `pivot_visualizations/TIMESTAMP/`:

```
pivot_visualizations/20250110_143022/
├── iteration_1_20250110_143022.jpg    # Candidates overlaid on image
├── iteration_1_20250110_143022.json   # Candidate metadata
├── iteration_2_20250110_143025.jpg
├── iteration_2_20250110_143025.json
├── iteration_3_20250110_143028.jpg
├── iteration_3_20250110_143028.json
└── final_result.json                  # Summary with selected trajectory
```

### Visualization Format

Each iteration image shows:
- **Colored circles**: Candidate waypoints (numbered 1-8)
- **Arrows**: Direction from camera center to each candidate
- **Numbers**: Trajectory IDs for VLM selection
- **Colors**: Unique per candidate (red, green, blue, yellow, etc.)

## Citation

If you use PIVOT in your research, please cite the original paper:

```bibtex
@article{google2024pivot,
    title={PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs},
    author={Nasiriany, Soroush and Xia, Fei and Yu, Wenhao and Xiao, Ted and Liang, Jacky and Dasgupta, Ishita and Xie, Annie and Driess, Danny and Wahid, Ayzaan and Xu, Zhuo and Vuong, Quan and Zhang, Tingnan and Lee, Tsang-Wei Edward and Lee, Kuang-Huei and Xu, Peng and Kirmani, Sean and Zhu, Yuke and Zeng, Andy and Hausman, Karol and Heess, Nicolas and Finn, Chelsea and Levine, Sergey and Ichter, Brian},
    year={2024},
    eprint={2402.07872},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Acknowledgments

- [See-Point-Fly](https://github.com/Hu-chih-yao/see-point-fly)
- [Microsoft AirSim](https://github.com/microsoft/AirSim) for simulation
- [PIVOT paper](https://arxiv.org/abs/2402.07872)
