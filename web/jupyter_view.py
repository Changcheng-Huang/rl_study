import streamlit as st
import os
import nbformat
from nbconvert import HTMLExporter
import urllib.parse
from algorithm_registry.integration import imported_algorithms

def show_jupyter_module():
    # Header section
    st.subheader("Jupyter Notebooks")
    st.caption("View implementation details or run the code directly in the cloud.")

    # --- Configuration ---
    # Replace these with your actual GitHub repository details
    GITHUB_USER = "sssonghuanan"
    GITHUB_REPO = "animations"
    GITHUB_BRANCH = "main"

    # Locate the notebook directory
    # Assumes structure: Animations-xxx/web/jupyter_view.py and Animations-xxx/notebook/
    notebook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebook")
    
    if not os.path.exists(notebook_dir):
        st.error(f"Notebook directory not found: {notebook_dir}")
        return

    # List built-in notebooks and notebooks supplied by installed packages.
    notebook_options = {
        f"Built-in · {file_name}": {
            "path": os.path.join(notebook_dir, file_name),
            "file_name": file_name,
            "builtin": True,
        }
        for file_name in sorted(
            f for f in os.listdir(notebook_dir) if f.endswith(".ipynb")
        )
    }
    for algorithm in imported_algorithms():
        if algorithm.manifest.notebook is None:
            continue
        file_name = os.path.basename(algorithm.manifest.notebook["file"])
        notebook_options[f"Imported · {algorithm.manifest.name}"] = {
            "path": str(algorithm.path / algorithm.manifest.notebook["file"]),
            "file_name": file_name,
            "builtin": False,
        }
    
    if not notebook_options:
        st.warning("No .ipynb files found in the directory.")
        return

    # Sidebar selection for the notebook
    selected_label = st.sidebar.selectbox("Select Notebook", list(notebook_options))
    selected = notebook_options[selected_label]
    selected_file = selected["file_name"]
    file_path = selected["path"]

    st.divider()

    # --- Visual Optimization: Aligning Download Button and Colab Badge ---
    
    # Use specific column ratios to prevent buttons from over-stretching
    btn_col1, btn_col2, _ = st.columns([0.25, 0.25, 0.5])

    with btn_col1:
        # Download button
        with open(file_path, "rb") as f:
            st.download_button(
                label="💾 Download File",
                data=f,
                file_name=selected_file,
                mime="application/x-ipynb+json",
                use_container_width=True
            )

    with btn_col2:
        # Generate Colab URL
        if selected["builtin"]:
            encoded_file = urllib.parse.quote(selected_file)
            colab_url = f"https://colab.research.google.com/github/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/notebook/{encoded_file}"

            # Optimization: Use Flexbox to force vertical alignment with Streamlit button height (38px)
            st.markdown(
                f"""
                <div style="display: flex; height: 38px; align-items: center;">
                    <a href="{colab_url}" target="_blank" style="text-decoration: none;">
                        <img src="https://colab.research.google.com/assets/colab-badge.svg"
                             alt="Open In Colab"
                             style="vertical-align: middle; border: 1px solid #ddd; border-radius: 4px;">
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.caption("Download the imported notebook to run it locally or in Colab.")

    # Status info
    st.info(f"Currently viewing: **{selected_file}**")

    # --- Notebook Rendering ---
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb_node = nbformat.read(f, as_version=4)
            
        html_exporter = HTMLExporter()
        # Remove input prompts for a cleaner look
        html_exporter.exclude_input_prompt = True 
        (body, resources) = html_exporter.from_notebook_node(nb_node)
        
        # Display the HTML content
        st.components.v1.html(body, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Error parsing notebook: {e}")
