import gradio as gr
import subprocess
import os
import tempfile
import shutil

def separate_music(audio_filepath):
    output_dir = tempfile.mkdtemp()
    
    command = [
        "python3", "-m", "demucs",
        "--out", output_dir,
        "--mp3",
        audio_filepath
    ]
    
    try:
        print(f"Running command: {' '.join(command)}")
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Demucs processing finished successfully.")
        

        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not subdirs:
            print("Error: Demucs created no output sub-directory.")
            return None, None, None, None
            
        model_output_folder = subdirs[0] 
 
        base_filename = os.path.basename(audio_filepath).rsplit('.', 1)[0]
        final_files_path = os.path.join(output_dir, model_output_folder, base_filename)
        
        print(f"Correctly constructed path for stems: {final_files_path}")

        vocals_path = os.path.join(final_files_path, "vocals.mp3")
        drums_path = os.path.join(final_files_path, "drums.mp3")
        bass_path = os.path.join(final_files_path, "bass.mp3")
        other_path = os.path.join(final_files_path, "other.mp3")

        return (
            vocals_path if os.path.exists(vocals_path) else None,
            drums_path if os.path.exists(drums_path) else None,
            bass_path if os.path.exists(bass_path) else None,
            other_path if os.path.exists(other_path) else None
        )

    except subprocess.CalledProcessError as e:
        print("Error during Demucs processing command.")
        print("Stderr:", e.stderr)
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected Python error occurred after processing: {e}")
        return None, None, None, None


demo = gr.Interface(
    fn=separate_music,
    inputs=gr.Audio(type="filepath", label="Upload Your Music"),
    outputs=[
        gr.Audio(label="Vocals"),
        gr.Audio(label="Drums"),
        gr.Audio(label="Bass"),
        gr.Audio(label="Other Instruments")
    ],
    title="🎵 AI Music Studio: Source Separation",
    description="Upload an audio file to separate it. This version uses the official Demucs library directly for maximum stability.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(debug=True)