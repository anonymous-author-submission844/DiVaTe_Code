
# Experiment of "Seperating the effects of prompt for text rendering"

This project supports generating images from prompts using multiple models: **FLUX**, **GPT/GPT-image-1/Gemini/SANA/Stable Diffusion3**, **Stable Diffusion**.

---

## ✨ How to Use
=======

### FLUX

1. Navigate to the flux experiment directory:
   ```bash
   cd exp/flux
   source .venv/bin/activate
   ```

2. Run batched generation:
   ```bash
   python run_flux_batch.py
   ```

---

### GPT / Gemini / SANA / GPT-image-1 / stable diffusion 3 / stable diffusion

1. From the root experiment folder:
   ```bash
   cd exp
   conda activate sana
   ```

2. Run main script with selected model:
   ```bash
   python main.py --model <model_name>
   ```

   Replace `<model_name>` with one of:
   - `gpt`
   - `gemini`
   - `sana`
   - `stable_diffusion3`
   - `gpt-image1`
   - `stable_diffusion`


## 🧪 Tip

Make sure all necessary environments are set up in advance:
- `.venv` for FLUX
- `conda` env `sana` for Gemini/GPT/SANA/GPT-image-1/Stable Diffusion 3/Stable Diffusion
---