import torch
import argparse
import torchaudio
from src.helpers import utils
from src.training import eval
import json
import importlib
import os
import yaml

config_path = "experiments/dc_waveformer/config.json"

def load_model(checkpoint_path, model_module="src.training.dcc_tf"):
    """ Load the trained model dynamically """

    if not os.path.exists(config_path) or os.stat(config_path).st_size == 0:
        raise ValueError(f"Error: Config file '{config_path}' is missing or empty!")

    # Load config safely
    with open(config_path, 'r') as f:
        try:
            config = json.load(f)  # Ensure valid JSON
        except json.JSONDecodeError:
            raise ValueError(f"Error: '{config_path}' contains invalid JSON!")

    model_params = config["model_params"]

    network = importlib.import_module(model_module)  # Dynamically import the model
    model = network.Net(**model_params)
    utils.load_checkpoint(checkpoint_path, model, data_parallel=False)
    model.eval()
    return model



def load_sound_classes(class_file="data/Classes.yaml"):
    """ Load available sound classes from `Classes.yaml`, ensuring all names are lowercase. """
    import yaml

    with open(class_file, 'r') as f:
        yaml_data = yaml.safe_load(f)  # Load YAML file

    # Convert all keys and values to lowercase, and split multi-word synonyms
    classes = {}
    for class_name, synonyms in yaml_data.items():
        class_name = class_name.lower()
        cleaned_synonyms = []

        for synonym in synonyms:
            # Split by commas if multiple synonyms exist in a single string
            cleaned_synonyms.extend([s.strip().lower() for s in synonym.split(',')])

        classes[class_name] = cleaned_synonyms  # Store cleaned list

    print(f"📂 Loaded Sound Classes: {classes}")  # 🔍 DEBUG PRINT

    return classes



def separate_sound(input_audio, model, sound_class_name):
    """ Process the input audio through the model to extract the specified sound class. """
    waveform, sample_rate = torchaudio.load(input_audio)

    # Convert mono to stereo if needed
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Convert sound class name to tensor representation
    label_tensor = torch.zeros(20)  # Assuming 20-dimensional label representation
    
    # 🔥 FIX: Use the correct class mapping from Classes.yaml
    if sound_class_name.lower() in sound_classes:
        mapped_class = sound_class_name.lower()  # Use the original key from YAML
    else:
        raise ValueError(f"❌ Error: '{sound_class_name}' not found in sound classes!")

    label_index = list(sound_classes.keys()).index(mapped_class)  # Get correct index
    label_tensor[label_index] = 1  # One-hot encoding for the class

    # Prepare model input
    inputs = {"mixture": waveform.unsqueeze(0), "label_vector": label_tensor.unsqueeze(0)}

    with torch.no_grad():
        separated_audio = model(inputs)  # Model expects a dictionary

    # Save output
    output_file = "separated_output.wav"
    torchaudio.save(output_file, separated_audio['x'].squeeze(0), sample_rate)
    print(f"🎉 Separated audio saved as '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--focus", type=str, required=True, help="Sound category to extract (e.g., 'dog barking')")
    parser.add_argument("--checkpoint", type=str, default="experiments/dc_waveformer/39.pt", help="Path to model checkpoint")
    parser.add_argument("--class_file", type=str, default="data/Classes.yaml", help="Path to the sound classes file")
    args = parser.parse_args()

    # Load model
    # model = load_model(args.checkpoint, "src.training.dcc_tf")
    model = load_model(args.checkpoint, "src.training.dcc_tf_binaural")

    # Load sound classes
    sound_classes = load_sound_classes(args.class_file)

    # Convert focus keyword to class index
    found = None
    for class_name, synonyms in sound_classes.items():
        if args.focus.lower() == class_name or args.focus.lower() in synonyms:
            found = class_name
            break

    if found is None:
        print(f"❌ Error: '{args.focus}' is not a valid sound class.")
        print(f"Available classes: {list(sound_classes.keys())}")
        exit(1)

    sound_class = found  # Assign the correct class name

    # Perform sound separation
    separate_sound(args.input, model, sound_class)