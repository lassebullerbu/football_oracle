import os
from src.load_data import load_data
from train import build_model, train_model

def run_pipeline(update_data=False):
    """
    all_in_one_pipeline(update_data=False)
    """

    # 1. load raw data and process to selected_data.csv if update_data=False it passes the step
    if update_data or not os.path.exists("./raw_data/selected_data.csv"):
        print("\n--- 🏁 Step 1: Processing Raw Data ---")
        load_data()
    else:
        print("\n--- ⏩ Step 1: Skipping Data Processing (File exists) ---")

    # 2. build model architecture
    print("\n--- 🏗️ Step 2: Building Model Architecture ---")
    model_structure = build_model()

    # 3. Train and save Pipeline/Model to models/
    print("\n--- 🧠 Step 3: Training and Saving Assets ---")
    trained_model = train_model(model_structure)

    print("\n" + "="*30)
    print("✅ SYSTEM READY FOR STREAMLIT!")
    print("="*30)

if __name__ == "__main__":
    # run new pipeline (update_data=True) if you want to reprocess raw data and retrain model
    # if you modify src/ change update_data=True
    run_pipeline(update_data=False)
