from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo("sumaiyaamir/agentic_rag_poisoning", exist_ok=True)

api.upload_large_folder(
    folder_path=r"C:\Users\Sumaiya Chan\OneDrive\Desktop\agentic_rag_poisoning",
    repo_id="sumaiyaamir/agentic_rag_poisoning",
    repo_type="model",
    num_workers=4,
    ignore_patterns=[
        "venc/*",
        "*.pyc",
        "__pycache__/*",
    ],
)

print(" Done! https://huggingface.co/sumaiyaamir/agentic_rag_poisoning")