import os
import shutil
if not os.getenv("QUARTO_PROJECT_RENDER_ALL"):
  exit()

# Import folders:
rel_dir = "../blog/docs"
target_dir = os.path.join(os.getenv("QUARTO_PROJECT_OUTPUT_DIR"), "blog")
if os.path.isdir(rel_dir):
  shutil.copytree(rel_dir, target_dir)
else:
  raise Exception("Folder not available.")

