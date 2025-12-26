#@title Batch Prediction
import os
from pathlib import Path
from tqdm.notebook import tqdm
from colabfold.batch import get_queries, run, set_model_type
from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
import matplotlib.pyplot as plt

# --- Input/Output Settings ---
input_dir = "/content/drive/MyDrive/input_sequences" #@param {type:"string"}
output_dir = "/content/drive/MyDrive/output_predictions" #@param {type:"string"}

# --- Run Settings ---
model_type = "auto" #@param ["auto", "alphafold2_ptm", "alphafold2_multimer_v1", "alphafold2_multimer_v2", "alphafold2_multimer_v3", "deepfold_v1", "alphafold2"]
num_recycles = "3" #@param ["auto", "0", "1", "3", "6", "12", "24", "48"]
recycle_early_stop_tolerance = "auto" #@param ["auto", "0.0", "0.5", "1.0"]
relax_max_iterations = 200 #@param [0, 200, 2000] {type:"raw"}
pairing_strategy = "greedy" #@param ["greedy", "complete"] {type:"string"}
calc_extra_ptm = False #@param {type:"boolean"}
max_msa = "auto" #@param ["auto", "512:1024", "256:512", "64:128", "32:64", "16:32"]
num_seeds = 1 #@param [1,2,4,8,16] {type:"raw"}
use_dropout = False #@param {type:"boolean"}
use_amber = False #@param {type:"boolean"}
use_templates = False #@param {type:"boolean"}
msa_mode = "mmseqs2_uniref_env" #@param ["mmseqs2_uniref_env", "mmseqs2_uniref","single_sequence","custom"]
pair_mode = "unpaired_paired" #@param ["unpaired_paired","paired","unpaired"] {type:"string"}
save_all = False #@param {type:"boolean"}
save_recycles = False #@param {type:"boolean"}
dpi = 200 #@param {type:"integer"}
display_images = False #@param {type:"boolean"}

# --- Parsing parameters ---
num_recycles = None if num_recycles == "auto" else int(num_recycles)
recycle_early_stop_tolerance = None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)
if max_msa == "auto": max_msa = None
num_relax = 1 if use_amber else 0

# --- Mount Drive if needed ---
if input_dir.startswith("/content/drive") or output_dir.startswith("/content/drive"):
    from google.colab import drive
    if not os.path.isdir("/content/drive"):
        drive.mount("/content/drive")

# --- Collect input files ---
input_path = Path(input_dir)
files_to_process = []
file_extensions = [".csv", ".fasta", ".a3m", ".fa"]

if input_path.is_dir():
    for ext in file_extensions:
        files_to_process.extend(list(input_path.glob(f"*{ext}")))
else:
    print(f"Input directory {input_dir} not found.")

# --- Callbacks ---
def input_features_callback(input_features):
  if display_images:
    from colabfold.plot import plot_msa_v2
    plot_msa_v2(input_features)
    plt.show()
    plt.close()

def prediction_callback(protein_obj, length,
                        prediction_result, input_features, mode):
  model_name, relaxed = mode
  if not relaxed:
    if display_images:
      from colabfold.colabfold import plot_protein
      fig = plot_protein(protein_obj, Ls=length, dpi=150)
      plt.show()
      plt.close()

# --- Processing Loop ---
# Using tqdm.notebook for nicer progress bars in Colab
for file_path in tqdm(files_to_process, desc="Processing Files"):
    jobname = file_path.stem
    # sanitize jobname
    import re
    jobname = re.sub(r'\W+', '', jobname)

    current_output_dir = os.path.join(output_dir, jobname)
    os.makedirs(current_output_dir, exist_ok=True)

    # Logging
    log_filename = os.path.join(current_output_dir, "log.txt")
    setup_logging(Path(log_filename))

    print(f"Running prediction for {file_path.name} -> {current_output_dir}")

    # Get queries
    queries, is_complex = get_queries(str(file_path))

    # Determine model type
    current_model_type = set_model_type(is_complex, model_type)

    # use_cluster_profile logic
    if "multimer" in current_model_type and max_msa is not None:
        use_cluster_profile = False
    else:
        use_cluster_profile = True

    # Download params if needed
    download_alphafold_params(current_model_type, Path("."))

    # Run prediction
    try:
        run(
            queries=queries,
            result_dir=current_output_dir,
            use_templates=use_templates,
            custom_template_path=None,
            num_relax=num_relax,
            msa_mode=msa_mode,
            model_type=current_model_type,
            num_models=5,
            num_recycles=num_recycles,
            relax_max_iterations=relax_max_iterations,
            recycle_early_stop_tolerance=recycle_early_stop_tolerance,
            num_seeds=num_seeds,
            use_dropout=use_dropout,
            model_order=[1,2,3,4,5],
            is_complex=is_complex,
            data_dir=Path("."),
            keep_existing_results=True,
            rank_by="auto",
            pair_mode=pair_mode,
            pairing_strategy=pairing_strategy,
            stop_at_score=float(100),
            prediction_callback=prediction_callback,
            dpi=dpi,
            zip_results=False,
            save_all=save_all,
            max_msa=max_msa,
            use_cluster_profile=use_cluster_profile,
            input_features_callback=input_features_callback,
            save_recycles=save_recycles,
            user_agent="colabfold/google-colab-batch",
            calc_extra_ptm=calc_extra_ptm,
        )
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

print("Batch processing complete.")
