import streamlit as st
import sys
import os

# Make the web package importable both through `streamlit run web/app.py` and
# Streamlit's AppTest runner.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
for import_path in (current_dir, root_dir):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from home import show_home
from animations import show_animation_library
from experiments import run_experiments_module
from notebooks import show_notebook_module
# 导入新模块
from jupyter_view import show_jupyter_module 
from admin import show_algorithm_admin

st.set_page_config(page_title='RL Education Platform', layout='wide', page_icon='')

st.sidebar.header('Platform Navigation')

if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = 'Home'

# 在 options 中新增 'Jupyter Notebooks'
app_mode = st.sidebar.radio(
    label='Platform section',
    options=[
        'Home',
        'Animation',
        'Theory',
        'Jupyter Notebooks',
        'RL Laboratory',
        'Manage Algorithms',
    ],
    key='app_mode',
    label_visibility='collapsed',
)

st.sidebar.divider()

if app_mode == 'Home':
    show_home()
elif app_mode == 'Animation':
    show_animation_library()
elif app_mode == 'Theory':
    show_notebook_module()
elif app_mode == 'Jupyter Notebooks': 
    show_jupyter_module()
elif app_mode == 'RL Laboratory':
    run_experiments_module()
elif app_mode == 'Manage Algorithms':
    show_algorithm_admin()

st.sidebar.info('Developed by Song Huanan')
